"""Analytical external-potential solver (UCSB variant), standalone.

This solver computes the external potential Φ(r) analytically from a provided
top-gate voltage map V_t(r) and a scalar back-gate voltage V_B, without reading
any potential data from files and without importing other solver implementations.

Analytical relation used in Fourier space (Φ in volts):

    Φ(q) = V_t(q) * sinh(β d_b |q|) / sinh(β (d_t + d_b) |q|)
    Φ_offset_real = - V_B * d_t / (d_t + d_b)   (spatially uniform, real space)

where β = sqrt(ε_parallel / ε_perp). The factor −e is applied later in the
energy term, so Φ here remains an electrostatic potential in volts.

Inputs:
- Either provide `Vt_grid` with shape (Nx, Ny) in volts on the chosen domain,
  or leave it as None for a zero top-gate (only back-gate offset is used).
- Domain bounds are set via x/y min/max in nanometers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Any
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d


# -----------------------------------------------------------------------------
# 1) Configuration – adds domain bounds, back-gate voltage, and V_t map
# -----------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Configuration parameters for the UCSB Thomas–Fermi simulation."""

    # Fundamental physical constants (SI)
    e: float = 1.602e-19  # Elementary charge [C]
    epsilon_0: float = 8.854e-12  # Vacuum permittivity [F/m]
    h: float = 6.62607015e-34  # Planck constant [J·s]

    # Experimental / material parameters
    B: float = 13.0  # Magnetic field [T]
    dt: float = 30e-9  # Distance to top-gate [m]
    db: float = 30e-9  # Distance to back-gate [m]

    epsilon_perp: float = 3.0
    epsilon_parallel: float = 6.6

    # Exchange–correlation data (nu, Exc)
    exc_file: str = "data/0-data/Exc_data_new.csv"
    exc_scale: float = 1.0

    # Optimisation parameters
    niter: int = 5
    step_size: float = 0.5
    lbfgs_maxiter: int = 1000
    lbfgs_maxfun: int = 100000

    # Grid size
    Nx: int = 64
    Ny: int = 64

    # Domain bounds (nm) – default square centered at 0
    x_min_nm: float = -150.0
    x_max_nm: float = 150.0
    y_min_nm: float = -150.0
    y_max_nm: float = 150.0

    # External top/back gate settings
    V_B: float = 0.0
    Vt_grid: Optional[np.ndarray] = None

    # Additional potential scaling/offset (kept for flexibility)
    potential_scale: float = 1.0
    potential_offset: float = 0.0

    # Identification
    solver_type: str = "solverUCSB"


# -----------------------------------------------------------------------------
# 2) Solver – overrides grid/potential preparation to use analytic Φ
# -----------------------------------------------------------------------------


class ThomasFermiSolver:
    """Thomas–Fermi solver using an analytic external potential Φ(r).

    Standalone implementation (no inheritance). It:
    - Builds the grid from explicit domain bounds
    - Computes Φ via FFT from V_t and adds a uniform back-gate offset
    - Prepares Coulomb and smoothing kernels
    - Optimises the energy functional via basinhopping + L-BFGS-B
    """

    J_to_meV = 1.0 / (1.602e-22)

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.Nx = int(cfg.Nx)
        self.Ny = int(cfg.Ny)
        self._prepare_grid()
        self._prepare_kernels()
        self._prepare_exc_table()
        self._init_density()

    # ------------------------------------------------------------------
    # Override: skip file I/O – nothing to load for the potential
    # ------------------------------------------------------------------
    # No external file loading required for UCSB; Φ is computed analytically

    # ------------------------------------------------------------------
    # Override: construct grid from bounds and compute Φ analytically
    # ------------------------------------------------------------------
    def _prepare_grid(self) -> None:
        cfg: SimulationConfig = self.cfg

        # Domain (convert nm → m)
        self.x_min = cfg.x_min_nm * 1e-9
        self.x_max = cfg.x_max_nm * 1e-9
        self.y_min = cfg.y_min_nm * 1e-9
        self.y_max = cfg.y_max_nm * 1e-9

        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dA = self.dx * self.dy
        self.A_total = self.Lx * self.Ly

        # Real-space grid
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.y = np.linspace(self.y_min, self.y_max, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Top-gate map V_t(r) [V]
        if cfg.Vt_grid is None:
            Vt = np.zeros((self.Nx, self.Ny), dtype=float)
        else:
            Vt = np.asarray(cfg.Vt_grid, dtype=float)
            if Vt.shape != (self.Nx, self.Ny):
                raise ValueError(
                    f"Vt_grid must have shape (Nx, Ny)=({self.Nx}, {self.Ny}), got {Vt.shape}"
                )

        # Compute Φ from V_t via FFT using the given screening relation
        # Create |q| grid
        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        q = np.sqrt(KX**2 + KY**2)

        # Material anisotropy factor β
        beta = np.sqrt(cfg.epsilon_parallel / cfg.epsilon_perp)

        # Ratio R(q) = sinh(β d_b |q|) / sinh(β (d_t + d_b) |q|)
        # Handle q→0 analytically: limit = db / (dt + db)
        dt = cfg.dt
        db = cfg.db
        Rq = np.empty_like(q)
        zero_mask = (q == 0)
        Rq[zero_mask] = db / (dt + db)
        nonzero = ~zero_mask
        if np.any(nonzero):
            denom_nz = np.sinh(beta * (dt + db) * q[nonzero])
            numer_nz = np.sinh(beta * db * q[nonzero])
            Rq[nonzero] = numer_nz / denom_nz

        # Back-gate uniform offset in real space → only q=0 (DC component)
        phi_bottom = cfg.V_B * dt / (dt + db)

        # FFT of Vt and apply the transfer function
        Vt_q = fft2(Vt)
        Phi_q = Vt_q * Rq
        # Add DC component corresponding to uniform real-space offset
        #Phi_q[0, 0] += phi_bottom * (self.Nx * self.Ny)

        # Inverse FFT to get Φ(r) in real space
        Phi_grid = np.real(ifft2(Phi_q))+phi_bottom

        # Apply any additional scaling/offset from config (kept for consistency)
        Phi_grid = Phi_grid * cfg.potential_scale + cfg.potential_offset

        self.Phi = Phi_grid

    # ------------------------------------------------------------------
    # Kernels and tables
    # ------------------------------------------------------------------
    def _prepare_kernels(self) -> None:
        cfg = self.cfg
        # Derived constants
        self.D = cfg.e * cfg.B / cfg.h  # Landau degeneracy [m-2]
        self.ell_B = np.sqrt(cfg.h / (2.0 * np.pi * cfg.e * cfg.B))

        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        q = np.sqrt(KX ** 2 + KY ** 2)

        # Gaussian form factor for magnetic length smoothing
        self.G_q = np.exp(-0.5 * (self.ell_B * q) ** 2)

        # Coulomb kernel in Fourier space (top/back gate screening)
        epsilon_hBN = np.sqrt(cfg.epsilon_perp * cfg.epsilon_parallel)
        beta = np.sqrt(cfg.epsilon_parallel / cfg.epsilon_perp)
        self.Vq = np.empty_like(q)
        nonzero = (q != 0)
        if np.any(nonzero):
            self.Vq[nonzero] = (
                cfg.e ** 2
                / (4.0 * np.pi * cfg.epsilon_0 * epsilon_hBN)
                * (4.0 * np.pi * np.sinh(beta * cfg.dt * q[nonzero]) * np.sinh(beta * cfg.db * q[nonzero]))
                / (np.sinh(beta * (cfg.dt + cfg.db) * q[nonzero]) * q[nonzero])
            )
        # q→0 analytical limit: Vq(0) = e^2/(ε0 ε_hBN) * β dt db / (dt+db)
        Vq_zero = (
            (cfg.e ** 2) / (cfg.epsilon_0 * epsilon_hBN)
            * beta * cfg.dt * cfg.db / (cfg.dt + cfg.db)
        )
        self.Vq[~nonzero] = Vq_zero

    def _prepare_exc_table(self) -> None:
        exc_data = np.loadtxt(self.cfg.exc_file, delimiter=",", skiprows=1)
        n_exc, Exc_vals = exc_data[:, 0], exc_data[:, 1]
        self.exc_interp = interp1d(
            n_exc,
            Exc_vals,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _init_density(self) -> None:
        # Classical (Thomas-Fermi) initial filling factor guess in [0,1]
        nu0 = np.clip(0.5 - 1.0 * (self.Phi - np.median(self.Phi)), 0.0, 1.0)
        self.nu0 = nu0.flatten()

    # ------------------------------------------------------------------
    # Core numerics
    # ------------------------------------------------------------------
    def gaussian_convolve(self, arr: np.ndarray) -> np.ndarray:
        arr_q = fft2(arr)
        conv_q = arr_q * self.G_q
        conv = ifft2(conv_q)
        return np.real(np.asarray(conv))

    def energy(self, nu_flat: np.ndarray) -> float:
        nu = nu_flat.reshape((self.Nx, self.Ny))
        nu_eff = self.gaussian_convolve(nu)
        n_eff = nu_eff * self.D  # density [m-2]

        # Hartree (Coulomb) energy via FFT
        n_fft = fft2(n_eff)
        n_q = n_fft * self.dA
        E_C = 0.5 / self.A_total * np.sum(self.Vq * np.abs(n_q) ** 2) * self.J_to_meV

        # External potential energy – compute in Fourier space like Hartree term
        # (keep real-space version for reference):
        E_phi = np.sum(-self.cfg.e * self.Phi * n_eff) * self.dA * self.J_to_meV
        #Phi_q = fft2(self.Phi) * self.dA
        #E_phi = (-self.cfg.e) * (1.0 / self.A_total) * np.real(np.sum(np.conj(Phi_q) * n_q)) * self.J_to_meV

        # Exchange–correlation energy (interpolated)
        E_xc = float(np.sum(self.exc_interp(nu_eff)) * self.dA * self.D) * float(self.cfg.exc_scale)

        total = E_phi + E_xc + E_C
        return float(total)

    # ------------------------------------------------------------------
    # Optimisation / post-processing
    # ------------------------------------------------------------------
    def optimise(self):
        start_time = float(__import__("time").time())

        self.energy_history: list[float] = [self.energy(self.nu0.copy())]

        def _bh_callback(x, f, accept):
            if accept:
                self.energy_history.append(float(f))

        #bounds = [(0.0, 1.0)] * (self.Nx * self.Ny)
        result = basinhopping(
            self.energy,
            self.nu0.copy(),
            minimizer_kwargs={
                "method": "L-BFGS-B",
                #"bounds": bounds,
                "options": {
                    "maxiter": self.cfg.lbfgs_maxiter,
                    "maxfun": self.cfg.lbfgs_maxfun,
                    "ftol": 1e-6,
                    "eps": 1e-8,
                },
            },
            niter=self.cfg.niter,
            stepsize=self.cfg.step_size,
            disp=True,
            callback=_bh_callback,
        )

        end_time = float(__import__("time").time())
        self.execution_time = end_time - start_time

        self.nu_opt = result.x.reshape((self.Nx, self.Ny))
        self.nu_smoothed = self.gaussian_convolve(self.nu_opt)
        self.optimisation_result = result
        return result

    # ------------------------------------------------------------------
    # Override plotting: use magnetic-length units (l_B) on axes
    # ------------------------------------------------------------------
    def plot_results(
        self,
        save_dir: Union[Path, str, None] = None,
        *,
        show: bool = False,
        title_extra: str = "",
    ):
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before plotting results.")

        from pathlib import Path as _Path
        import matplotlib.pyplot as _plt

        # Extent in units of l_B
        extent_lB = (
            self.x_min / self.ell_B,
            self.x_max / self.ell_B,
            self.y_min / self.ell_B,
            self.y_max / self.ell_B,
        )

        save_dir_path = _Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)

        # ν(r)
        fig1 = _plt.figure(figsize=(6, 5))
        _plt.imshow(
            self.nu_smoothed.T,
            extent=extent_lB,
            origin="lower",
            cmap="inferno",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        base_title = "Optimised Filling Factor ν(r)"
        if title_extra:
            base_title += f"\n{title_extra}"
        _plt.title(base_title)
        _plt.xlabel("x [l_B]")
        _plt.ylabel("y [l_B]")
        _plt.colorbar(label="ν")
        _plt.tight_layout()
        if save_dir_path is not None:
            fig1.savefig(save_dir_path / "nu_smoothed.png", dpi=300)

        # Φ(r)
        fig2 = _plt.figure(figsize=(6, 5))
        _plt.imshow(
            self.Phi.T,
            extent=extent_lB,
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        base_title2 = "External Potential Φ(r) [V]"
        if title_extra:
            base_title2 += f"\n{title_extra}"
        _plt.title(base_title2)
        _plt.xlabel("x [l_B]")
        _plt.ylabel("y [l_B]")
        _plt.colorbar(label="Φ [V]")
        _plt.tight_layout()
        if save_dir_path is not None:
            fig2.savefig(save_dir_path / "phi.png", dpi=300)

        # Energy history plot (re-use parent's helper)
        if save_dir_path is not None and hasattr(self, "energy_history") and len(self.energy_history) >= 2:
            self._plot_energy_decrease(save_dir_path)

        # --------------------------------------------------------------
        # NEW: cross-section plot along x (y ≈ 0): Φ_ext [meV] vs x/ℓ_B and ν
        # --------------------------------------------------------------
        import numpy as _np
        y_index = int(_np.argmin(_np.abs(self.y)))
        x_over_lB = self.x / self.ell_B
        # External potential energy for electron: Φ_ext = e Φ, in meV
        phi_ext_meV = (self.cfg.e * self.Phi[:, y_index]) * self.J_to_meV
        nu_line = self.nu_smoothed[:, y_index]

        fig3, ax1 = _plt.subplots(figsize=(6.2, 4.5))
        color1 = "tab:red"
        ln1 = ax1.plot(x_over_lB, phi_ext_meV, color=color1, linewidth=2.2, label="Φ_ext [meV]")
        ax1.set_xlabel("x [ℓ_B]")
        ax1.set_ylabel("Φ_ext [meV]", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:blue"
        ln2 = ax2.plot(x_over_lB, nu_line, color=color2, linewidth=2.0, label="ν")
        ax2.set_ylabel("ν", color=color2)
        ax2.set_ylim(0.0, 1.0)
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        fig3.legend(lines, labels, loc="upper right")
        _plt.tight_layout()
        if save_dir_path is not None:
            fig3.savefig(save_dir_path / "phi_nu_cross_section.png", dpi=300)
            # Save raw arrays for reproducibility
            _np.savez_compressed(
                save_dir_path / "phi_nu_cross_section.npz",
                x_over_lB=x_over_lB,
                phi_ext_meV=phi_ext_meV,
                nu_line=nu_line,
                y_index=y_index,
            )

        if show:
            _plt.show()
        else:
            _plt.close(fig1)
            _plt.close(fig2)
            _plt.close(fig3)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_results(self, output_dir: Union[Path, str] = "results") -> None:
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before saving results.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save arrays – include energy history if available
        save_dict: dict[str, Any] = {
            "nu_opt": self.nu_opt,
            "nu_smoothed": self.nu_smoothed,
            "Phi": self.Phi,
            "x": self.x,
            "y": self.y,
        }
        if hasattr(self, "energy_history") and self.energy_history:
            save_dict["energy_history"] = np.array(self.energy_history)

        np.savez_compressed(out_path / "results.npz", **save_dict)

        # Save optimisation summary
        with open(out_path / "optimisation.txt", "w", encoding="utf-8") as f:
            f.write(str(self.optimisation_result))

        # Save full simulation configuration
        try:
            from dataclasses import asdict as _asdict
            cfg_dict = _asdict(self.cfg)
        except Exception:
            cfg_dict = {k: getattr(self.cfg, k) for k in dir(self.cfg) if not k.startswith("__") and not callable(getattr(self.cfg, k))}

        with open(out_path / "simulation_parameters.txt", "w", encoding="utf-8") as f:
            for key, val in cfg_dict.items():
                f.write(f"{key} = {val}\n")
            if hasattr(self, "execution_time"):
                f.write(f"\n# Execution time\n")
                f.write(f"execution_time_seconds = {self.execution_time:.6f}\n")
                f.write(f"execution_time_minutes = {self.execution_time/60:.6f}\n")

    # ------------------------------------------------------------------
    # Internal helper: energy vs iteration / ΔE plot
    # ------------------------------------------------------------------
    def _plot_energy_decrease(self, out_dir: Path) -> None:
        import numpy as _np
        import matplotlib.pyplot as _plt

        energy_full = _np.asarray(self.energy_history, dtype=float)
        if energy_full.size < 2:
            return

        iterations = _np.arange(1, energy_full.size)
        energy = energy_full[1:]
        delta_E = -_np.diff(energy_full)

        fig, ax1 = _plt.subplots(figsize=(7, 4))
        ax1.set_xlabel("Basinhopping iteration")
        ax1.set_ylabel("Energy [meV]", color="tab:blue")
        ax1.plot(iterations, energy, marker="o", color="tab:blue", label="Energy")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Energy decrease [meV]", color="tab:red")
        ax2.bar(iterations[1:], delta_E[1:], color="tab:red", alpha=0.4, label="ΔEnergy (iter≥2)")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        fig.tight_layout()

        lines, labels = [], []
        l, lab = ax1.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
        l2, lab2 = ax2.get_legend_handles_labels()
        lines.extend(l2)
        labels.extend(lab2)
        fig.legend(lines, labels, loc="upper right")

        fig.savefig(out_dir / "energy_decrease.png", dpi=300, bbox_inches="tight")
        _plt.close(fig)


# Convenience aliases (match repository style)
SimulationConfigUCSB = SimulationConfig
ThomasFermiSolverUCSB = ThomasFermiSolver



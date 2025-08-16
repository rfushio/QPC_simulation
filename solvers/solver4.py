"""Thomas–Fermi solver (solver4): external potential from data, self-contained.

This solver mirrors the numerics and plotting flow of the UCSB analytic solver,
but replaces the analytic Φ construction with loading/interpolating Φ from a
text/CSV file (or in-memory arrays). Axes use magnetic-length units in plots.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union, Any
from pathlib import Path

import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# Configuration – includes external potential file loading
# -----------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    # Physical constants
    e: float = 1.602e-19
    epsilon_0: float = 8.854e-12
    h: float = 6.62607015e-34

    # Experimental / material parameters
    B: float = 13.0
    dt: float = 30e-9
    db: float = 30e-9

    epsilon_perp: float = 3.0
    epsilon_parallel: float = 6.6

    # Exchange–correlation
    exc_file: str = "data/0-data/Exc_data_new.csv"
    exc_scale: float = 1.0

    # Optimisation
    niter: int = 5
    step_size: float = 0.5
    lbfgs_maxiter: int = 1000
    lbfgs_maxfun: int = 100000

    # Grid
    Nx: int = 64
    Ny: int = 64

    # External potential source (required): text/CSV file with columns x_nm, y_nm, Phi[V]
    potential_file: Optional[str] = None
    margin: float = 0.0  # extra margin around data bounds [m]

    # Additional scaling/offset
    potential_scale: float = 1.0
    potential_offset: float = 0.0

    # Identification
    solver_type: str = "solver4"

    # Matryoshka (progressive refinement)
    use_matryoshka: bool = False
    matryoshka_min_N: int = 32
    coarse_accept_limit: int = 1


class ThomasFermiSolver:
    """Thomas–Fermi solver using an external potential Φ loaded from data.

    Follows the same kernels, energy, and plotting conventions as the UCSB
    analytic solver, but without computing Φ from Vt.
    """

    J_to_meV = 1.0 / (1.602e-22)

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.Nx = int(cfg.Nx)
        self.Ny = int(cfg.Ny)
        self._load_external_potential()
        self._prepare_grid()
        self._prepare_kernels()
        self._prepare_exc_table()
        self._init_density()
        setattr(self, "_bh_step", 0)

    # ------------------------------------------------------------------
    # External potential loading and grid preparation
    # ------------------------------------------------------------------
    def _load_external_potential(self) -> None:
        cfg = self.cfg
        if not cfg.potential_file:
            raise ValueError("Either potential_data or potential_file must be provided.")

        data = np.loadtxt(cfg.potential_file, comments="%")
        if data.shape[1] < 3:
            raise ValueError("Potential file must have ≥3 columns (x_nm, y_nm, Phi[V]).")
        self._x_data = data[:, 0] * 1e-9
        self._y_data = data[:, 1] * 1e-9
        self._phi_data = data[:, -1]

    def _prepare_grid(self) -> None:
        cfg = self.cfg

        # Domain: 40×ℓ_B square centered at 0 (±20 ℓ_B), like mainUCSB
        ell_B = np.sqrt(cfg.h / (2.0 * np.pi * cfg.e * cfg.B))
        half_span = 20.0 * ell_B
        self.x_min, self.x_max = -half_span, +half_span
        self.y_min, self.y_max = -half_span, +half_span

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

        # Nearest-neighbour assignment of Φ from scattered data to regular grid
        # Build KD-tree on data points and query nearest for each grid node
        data_points = np.column_stack((self._x_data, self._y_data))
        tree = cKDTree(data_points)
        grid_points = np.column_stack((self.X.ravel(), self.Y.ravel()))
        _, idx = tree.query(grid_points, k=1)
        Phi_grid = self._phi_data[idx].reshape(self.Nx, self.Ny)

        # Scale and offset
        Phi_grid = Phi_grid * cfg.potential_scale + cfg.potential_offset
        self.Phi = np.asarray(Phi_grid, dtype=float)

    def _prepare_kernels(self) -> None:
        cfg = self.cfg
        self.D = cfg.e * cfg.B / cfg.h
        self.ell_B = np.sqrt(cfg.h / (2.0 * np.pi * cfg.e * cfg.B))

        kx = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        q = np.sqrt(KX ** 2 + KY ** 2)

        # Gaussian form factor (matches UCSB smoothing)
        self.G_q = np.exp(-0.5 * (self.ell_B * q) ** 2)

        # Coulomb kernel with top/back screening
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
        # q→0 limit
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
        nu0 = np.clip(0.5 - 1.0 * (self.Phi - np.median(self.Phi)), 0.0, 1.0)
        self.nu0 = nu0.flatten()

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------
    def gaussian_convolve(self, arr: np.ndarray) -> np.ndarray:
        arr_q = fft2(arr)
        conv_q = arr_q * self.G_q
        conv = ifft2(conv_q)
        return np.real(np.asarray(conv))

    def energy(self, nu_flat: np.ndarray) -> float:
        nu = nu_flat.reshape((self.Nx, self.Ny))
        nu_eff = self.gaussian_convolve(nu)
        n_eff = nu_eff * self.D

        # Hartree via FFT
        n_fft = fft2(n_eff)
        n_q = n_fft * self.dA
        E_C = 0.5 / self.A_total * np.sum(self.Vq * np.abs(n_q) ** 2) * self.J_to_meV

        # External potential energy (real-space form)
        E_phi = np.sum(-self.cfg.e * self.Phi * n_eff) * self.dA * self.J_to_meV

        # Exchange–correlation
        E_xc = float(np.sum(self.exc_interp(nu_eff)) * self.dA * self.D) * float(self.cfg.exc_scale)

        return float(E_phi + E_xc + E_C)

    # ------------------------------------------------------------------
    # Optimisation / matryoshka
    # ------------------------------------------------------------------
    def optimise(self):
        if not getattr(self.cfg, "use_matryoshka", False):
            start_time = float(__import__("time").time())
            self.energy_history: list[float] = [self.energy(self.nu0.copy())]
            try:
                _total_iter = int(self.cfg.niter)
            except Exception:
                _total_iter = 0
            print(f"[solver4] start single-stage niter={_total_iter}")

            def _bh_callback(x, f, accept):
                step_i = getattr(self, "_bh_step", 0) + 1
                setattr(self, "_bh_step", step_i)
                if accept:
                    self.energy_history.append(float(f))
                print(f"[solver4] iter={step_i}/{_total_iter} accept={bool(accept)} energy={float(f):.6f} meV")

            result = basinhopping(
                self.energy,
                self.nu0.copy(),
                minimizer_kwargs={
                    "method": "L-BFGS-B",
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
            print("[solver4] finish single-stage")
            self.nu_opt = result.x.reshape((self.Nx, self.Ny))
            self.nu_smoothed = self.gaussian_convolve(self.nu_opt)
            self.optimisation_result = result
            return result

        # Matryoshka progressive refinement
        start_time = float(__import__("time").time())
        target_N: int = int(self.Nx)
        min_N: int = int(max(2, self.cfg.matryoshka_min_N))

        if target_N <= min_N:
            resolutions: list[int] = [target_N]
        else:
            resolutions = []
            n = min_N
            while n < target_N:
                resolutions.append(n)
                if n * 2 > target_N:
                    resolutions.append(target_N)
                    break
                n *= 2
            else:
                if resolutions[-1] != target_N:
                    resolutions.append(target_N)

        self.energy_history = []

        prev_nu: Optional[np.ndarray] = None
        prev_N: Optional[int] = None

        for res in resolutions:
            # Resample phi and initial guess
            rgi = RegularGridInterpolator((np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny)), self.Phi)
            tgt = np.linspace(0, 1, res)
            Xn, Yn = np.meshgrid(tgt, tgt, indexing="ij")
            phi_res = rgi(np.stack([Xn, Yn], axis=-1))

            stage_cfg = replace(
                self.cfg,
                Nx=int(res),
                Ny=int(res),
            )
            stage = ThomasFermiSolver(stage_cfg)
            stage.Phi = np.asarray(phi_res, dtype=float)
            stage._prepare_kernels()
            stage._prepare_exc_table()
            stage._init_density()

            if prev_nu is not None and prev_N is not None:
                rgi_nu = RegularGridInterpolator((np.linspace(0, 1, prev_N), np.linspace(0, 1, prev_N)), prev_nu)
                stage.nu0 = rgi_nu(np.stack([Xn, Yn], axis=-1)).flatten()

            stage.energy_history = [stage.energy(stage.nu0.copy())]

            accepted_count = 0
            _accepted_x: list[np.ndarray] = []

            class _EarlyStop(Exception):
                pass

            def _bh_callback(x, f, accept):
                nonlocal accepted_count
                # track per-stage step
                step_i = getattr(stage, "_bh_step", 0) + 1
                setattr(stage, "_bh_step", step_i)
                if accept:
                    accepted_count += 1
                    _accepted_x.append(x.copy())
                    stage.energy_history.append(float(f))
                # progress print
                try:
                    _stage_total = int(stage.cfg.niter)
                except Exception:
                    _stage_total = 0
                print(
                    f"[solver4] stage N={res} iter={step_i}/{_stage_total} accept={bool(accept)} energy={float(f):.6f} meV accepted={accepted_count}"
                )
                if (
                    res < target_N
                    and getattr(stage.cfg, "coarse_accept_limit", 1) > 0
                    and accepted_count >= getattr(stage.cfg, "coarse_accept_limit", 1)
                ):
                    print(f"[solver4] stage N={res} early-stop after {accepted_count} accepts")
                    raise _EarlyStop

            try:
                result = basinhopping(
                    stage.energy,
                    stage.nu0.copy(),
                    minimizer_kwargs={
                        "method": "L-BFGS-B",
                        "options": {
                            "maxiter": stage.cfg.lbfgs_maxiter,
                            "maxfun": stage.cfg.lbfgs_maxfun,
                            "ftol": 1e-6,
                            "eps": 1e-8,
                        },
                    },
                    niter=stage.cfg.niter,
                    stepsize=stage.cfg.step_size,
                    disp=False,
                    callback=_bh_callback,
                )
            except _EarlyStop:
                result = None

            best_x = result.x if result is not None else (_accepted_x[-1] if _accepted_x else stage.nu0.copy())

            stage.nu_opt = best_x.reshape((stage.Nx, stage.Ny))
            stage.nu_smoothed = stage.gaussian_convolve(stage.nu_opt)

            if res == target_N:
                self.energy_history.extend(stage.energy_history)
            else:
                self.energy_history.extend(stage.energy_history[:-1])

            prev_nu = stage.nu_opt
            prev_N = res

            if res == target_N:
                self.__dict__.update(stage.__dict__)
                self.optimisation_result = result if result is not None else {
                    "N": target_N,
                    "fun": float(self.energy(self.nu_opt.flatten())),
                }
                break

        end_time = float(__import__("time").time())
        self.execution_time = end_time - start_time
        return getattr(self, "optimisation_result")

    # ------------------------------------------------------------------
    # Plotting and persistence (match UCSB style)
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

        import matplotlib.pyplot as _plt

        extent_lB = (
            self.x_min / self.ell_B,
            self.x_max / self.ell_B,
            self.y_min / self.ell_B,
            self.y_max / self.ell_B,
        )

        save_path = Path(save_dir) if save_dir is not None else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        # ν(r)
        fig1 = _plt.figure(figsize=(6, 5))
        _plt.imshow(self.nu_smoothed.T, extent=extent_lB, origin="lower", cmap="inferno", aspect="auto")
        title = "Optimised Filling Factor ν(r)"
        if title_extra:
            title += f"  |  {title_extra}"
        _plt.title(title)
        _plt.xlabel("x [ℓ_B]")
        _plt.ylabel("y [ℓ_B]")
        _plt.colorbar(label="ν")
        _plt.tight_layout()
        if save_path is not None:
            fig1.savefig(save_path / "nu_smoothed.png", dpi=300)

        # Φ(r)
        fig2 = _plt.figure(figsize=(6, 5))
        _plt.imshow(self.Phi.T, extent=extent_lB, origin="lower", cmap="viridis", aspect="auto")
        title2 = "External Potential Φ(r) [V]"
        if title_extra:
            title2 += f"\n{title_extra}"
        _plt.title(title2)
        _plt.xlabel("x [ℓ_B]")
        _plt.ylabel("y [ℓ_B]")
        _plt.colorbar(label="Φ [V]")
        _plt.tight_layout()
        if save_path is not None:
            fig2.savefig(save_path / "phi.png", dpi=300)

        # Energy history plot
        if save_path is not None and hasattr(self, "energy_history") and len(self.energy_history) >= 2:
            self._plot_energy_decrease(save_path)

        # Cross-section plot
        import numpy as _np
        y_index = int(_np.argmin(_np.abs(self.y)))
        x_over_lB = self.x / self.ell_B
        phi_ext_meV = (self.cfg.e * self.Phi[:, y_index]) * self.J_to_meV
        nu_line = self.nu_smoothed[:, y_index]

        fig3, ax1 = _plt.subplots(figsize=(6.2, 4.5))
        ln1 = ax1.plot(x_over_lB, phi_ext_meV, color="tab:red", linewidth=2.2, label="Φ_ext [meV]")
        ax1.set_xlabel("x [ℓ_B]")
        ax1.set_ylabel("Φ_ext [meV]", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        ax2 = ax1.twinx()
        ln2 = ax2.plot(x_over_lB, nu_line, color="tab:blue", linewidth=2.0, label="ν")
        ax2.set_ylabel("ν", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        fig3.legend(lines, labels, loc="upper right")
        _plt.tight_layout()
        if save_path is not None:
            fig3.savefig(save_path / "phi_nu_cross_section.png", dpi=300)

        if show:
            _plt.show()
        else:
            _plt.close(fig1)
            _plt.close(fig2)
            _plt.close(fig3)

    def save_results(self, output_dir: Union[Path, str] = "results") -> None:
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before saving results.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
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

        # Save optimisation summary and configuration
        with open(out_path / "optimisation.txt", "w", encoding="utf-8") as f:
            f.write(str(getattr(self, "optimisation_result", {})))

        try:
            from dataclasses import asdict as _asdict
            cfg_dict = _asdict(self.cfg)
        except Exception:
            cfg_dict = {k: getattr(self.cfg, k) for k in dir(self.cfg) if not k.startswith("__") and not callable(getattr(self.cfg, k))}

        with open(out_path / "simulation_parameters.txt", "w", encoding="utf-8") as f:
            # Include potential pairing summary (file name)
            if getattr(self.cfg, "potential_file", None):
                f.write(f"potential_file = {self.cfg.potential_file}\n")
            # Dump remaining config (excluding large arrays)
            for key, val in cfg_dict.items():
                if key in {"potential_data"}:  # ensure not present, but guard anyway
                    continue
                f.write(f"{key} = {val}\n")
            if hasattr(self, "execution_time"):
                f.write(f"\n# Execution time\n")
                f.write(f"execution_time_seconds = {self.execution_time:.6f}\n")
                f.write(f"execution_time_minutes = {self.execution_time/60:.6f}\n")

    # Energy decrease plot with color-coded ΔE (green/red/gray)
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
        ax2.set_ylabel("Energy decrease [meV]")
        de_vals = delta_E[1:]
        x_vals = iterations[1:]
        colors = [("tab:green" if v > 0 else ("tab:red" if v < 0 else "0.6")) for v in de_vals]
        ax2.bar(x_vals, de_vals, color=colors, alpha=0.6)
        ax2.tick_params(axis="y")

        fig.tight_layout()

        lines, labels = [], []
        l, lab = ax1.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
        from matplotlib.patches import Patch as _Patch
        lines.append(_Patch(color="tab:green", alpha=0.6))
        labels.append("ΔE > 0 (decrease)")
        lines.append(_Patch(color="tab:red", alpha=0.6))
        labels.append("ΔE < 0 (increase)")
        fig.legend(lines, labels, loc="upper right")

        fig.savefig(out_dir / "energy_decrease.png", dpi=300, bbox_inches="tight")
        _plt.close(fig)


# Aliases
SimulationConfig4 = SimulationConfig
ThomasFermiSolver4 = ThomasFermiSolver



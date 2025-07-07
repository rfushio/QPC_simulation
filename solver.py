import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path

from scipy.fft import fft2, ifft2
from scipy.optimize import basinhopping
from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter


@dataclass
class SimulationConfig:
    """Configuration parameters for Thomas–Fermi simulation."""

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

    margin: float = 0.0  # Extra margin around data bounds [m]

    # External potential file (x [nm], y [nm], V [V])
    potential_file: str = "data/0-data/VNS=2.1.txt"
<<<<<<< HEAD
    potential_scale: float = 1.0  # Scale factor applied to Φ
    potential_offset: float = 0.0  # Constant offset added to Φ
=======
    potential_scale: float = 1  # Scale factor applied to Φ
    potential_offset: float = 0  # Constant offset added to Φ
>>>>>>> auto-simulation

    # Exchange–correlation data (nu, Exc)
    exc_file: str = "data/0-data/Exc_data_digitized.csv"

    # Optimisation parameters
    niter: int = 5  # Basinhopping outer iterations
    lbfgs_maxiter: int = 1000
    lbfgs_maxfun: int = 100000  # Maximum number of function evaluations for L-BFGS-B

    # Grid size (number of grid points in x and y directions)
    Nx: int = 64
    Ny: int = 64

    # Potential data number
    n_potentials: int = 0 #number of potentials to simulate


    # In-memory external potential data (x [nm], y [nm], V [V])
    potential_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


class ThomasFermiSolver:
    """Thomas–Fermi solver for 2D electron gas in magnetic field."""

    J_to_meV = 1.0 / (1.602e-22)  # Energy conversion factor (constant)

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg #cfg is a SimulationConfig object    
        self.Nx = cfg.Nx #Nx is the number of grid points in the x direction
        self.Ny = cfg.Ny #Ny is the number of grid points in the y direction
        self.n_potentials = cfg.n_potentials #n_potentials is the number of potentials to simulate
        self._load_external_potential()
        self._prepare_grid()
        self._prepare_kernels()
        self._prepare_exc_table()
        self._init_density()

    # ---------------------------------------------------------------------
    # Pre-processing helpers
    # ---------------------------------------------------------------------
    def _load_external_potential(self):
        """Load the external potential data from text file or in-memory arrays."""
        if self.cfg.potential_data is not None:
            # Unpack provided arrays (assumed in nm units for x, y)
            x_nm, y_nm, V_vals = self.cfg.potential_data
            self.x_data = np.asarray(x_nm) * 1e-9  # nm → m
            self.y_data = np.asarray(y_nm) * 1e-9  # nm → m
            self.V_data = np.asarray(V_vals)
            return

        # Fallback to reading from file
        data = np.loadtxt(self.cfg.potential_file, comments="%")
        
        # Support both 3-column (x y V) and 4-column (x y z V)
        if data.shape[1] < 3:
            raise ValueError("Potential file must have ≥3 columns (x, y, V).")
        
        self.x_data = data[:, 0] * 1e-9  # nm → m
        self.y_data = data[:, 1] * 1e-9  # nm → m
        self.V_data = data[:, -1]        # take last column as Φ

    def _prepare_grid(self):
        cfg = self.cfg
        # Simulation domain bounds with margin
        x_min, x_max = self.x_data.min() - cfg.margin, self.x_data.max() + cfg.margin
        y_min, y_max = self.y_data.min() - cfg.margin, self.y_data.max() + cfg.margin

        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

        self.Lx, self.Ly = (x_max - x_min), (y_max - y_min)
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        self.dA = self.dx * self.dy
        self.A_total = self.Lx * self.Ly

        # Real-space grid
        self.x = np.linspace(x_min, x_max, self.Nx)
        self.y = np.linspace(y_min, y_max, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # Interpolate Φ onto the grid
        Phi_grid = griddata(
            points=np.column_stack((self.x_data, self.y_data)),
            values=self.V_data,
            xi=(self.X, self.Y),
            method="linear",
        )

        # Fill NaNs with nearest
        mask = np.isnan(Phi_grid)
        if np.any(mask):
            Phi_grid[mask] = griddata(
                points=np.column_stack((self.x_data, self.y_data)),
                values=self.V_data,
                xi=(self.X[mask], self.Y[mask]),
                method="nearest",
            )

        # Apply scaling / offset
        Phi_grid *= cfg.potential_scale
        Phi_grid += cfg.potential_offset

        self.Phi = Phi_grid

    def _prepare_kernels(self):
        cfg = self.cfg
        # Derived constants
        self.D = cfg.e * cfg.B / cfg.h  # Landau degeneracy [m-2]
        self.ell_B = np.sqrt(cfg.h / (2 * np.pi * cfg.e * cfg.B))

        kx = 2 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.Ny, d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        q = np.sqrt(KX ** 2 + KY ** 2)
        q[0, 0] = 1e-20  # avoid division by zero

        # Gaussian form factor for magnetic length smoothing
        self.G_q = np.exp(-0.5 * (self.ell_B * q) ** 2)

        # Coulomb kernel in Fourier space (top/back gate screening)
        epsilon_hBN = np.sqrt(cfg.epsilon_perp * cfg.epsilon_parallel)
        beta = np.sqrt(cfg.epsilon_parallel / cfg.epsilon_perp)
        self.Vq = (
            self.cfg.e ** 2
            / (4 * np.pi * self.cfg.epsilon_0 * epsilon_hBN)
            * (4 * np.pi * np.sinh(beta * cfg.dt * q) * np.sinh(beta * cfg.db * q))
            / (np.sinh(beta * (cfg.dt + cfg.db) * q) * q)
        )

    def _prepare_exc_table(self):
        exc_data = np.loadtxt(self.cfg.exc_file, delimiter=",", skiprows=1)
        n_exc, Exc_vals = exc_data[:, 0], exc_data[:, 1]
        self.exc_interp = interp1d(n_exc, Exc_vals, kind="linear", )#fill_value="extrapolate"

    def _init_density(self):
        # Classical (Thomas-Fermi) initial filling factor guess in [0,1]
        nu0 = np.clip(0.5 - 1.0 * (self.Phi - np.median(self.Phi)), 0.0, 1.0)
        self.nu0 = nu0.flatten()

    # ---------------------------------------------------------------------
    # Core numerical routines
    # ---------------------------------------------------------------------
    def gaussian_convolve(self, arr: np.ndarray) -> np.ndarray:
        """Magnetic-length Gaussian smoothing with periodic BCs."""
        sigma = self.ell_B / self.dx  # convert ℓ_B to grid units
        return gaussian_filter(arr, sigma=sigma, mode="wrap")

    def energy(self, nu_flat: np.ndarray) -> float:
        """Total energy functional in meV."""
        nu = nu_flat.reshape((self.Nx, self.Ny))
        nu_eff = self.gaussian_convolve(nu)
        n_eff = nu_eff * self.D  # density [m-2]

        # Hartree (Coulomb) energy via FFT
        n_fft = fft2(n_eff)
        n_q = n_fft * self.dA
        E_C = 0.5 / self.A_total * np.sum(self.Vq * np.abs(n_q) ** 2) * self.J_to_meV

        # External potential energy
        E_phi = np.sum(-self.cfg.e * self.Phi * n_eff) * self.dA * self.J_to_meV

        # Exchange–correlation energy (interpolated)
        E_xc = np.sum(self.exc_interp(nu_eff)) * self.dA * self.D

        total = E_phi + E_xc + E_C
        return float(total)

    # ---------------------------------------------------------------------
    # Optimisation / post-processing
    # ---------------------------------------------------------------------
    def optimise(self):
        """Run global optimisation to find the ground-state filling factor."""
        bounds = [(0.0, 1.0)] * (self.Nx * self.Ny)
        result = basinhopping(
            self.energy,
            self.nu0.copy(),
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {
                    "maxiter": self.cfg.lbfgs_maxiter,
                    "maxfun": self.cfg.lbfgs_maxfun,
                    "ftol": 1e-6,
                    "eps": 1e-8,
                },
            },
            niter=self.cfg.niter,
            stepsize=0.01,
            disp=True,
        )

        self.nu_opt = result.x.reshape((self.Nx, self.Ny))
        self.nu_smoothed = self.gaussian_convolve(self.nu_opt)
        self.optimisation_result = result
        return result

    # ---------------------------------------------------------------------
    # Helper visualisation routines
    # ---------------------------------------------------------------------
    def plot_results(self, save_dir: Path | str | None = None, *, show: bool = False):
        """Visualise (and optionally save) results.

        Parameters
        ----------
        save_dir : Path | str | None
            Directory where figures are saved (PNG). If None, figures are not saved.
        show : bool, default False
            If True, display figures interactively via ``plt.show()`` (blocking).
            If False (default), figures are closed after saving so that batch
            scripts can proceed without manual intervention.
        """
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before plotting results.")

        # define as tuple for static type compatibility with matplotlib stubs
        extent = (self.x_min * 1e9, self.x_max * 1e9, self.y_min * 1e9, self.y_max * 1e9)

        # Ensure pathlib Path
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)

        # ν(r)
        fig1 = plt.figure(figsize=(6, 5))
        plt.imshow(self.nu_smoothed, extent=extent, origin="lower", cmap="inferno")
        plt.title("Optimised Filling Factor ν(r)")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.colorbar(label="ν")
        plt.tight_layout()
        if save_dir_path is not None:
            fig1.savefig(save_dir_path / "nu_smoothed.png", dpi=300)

        # Φ(r)
        fig2 = plt.figure(figsize=(6, 5))
        plt.imshow(self.Phi, extent=extent, origin="lower", cmap="viridis")
        plt.title("External Potential Φ(r) [V]")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.colorbar(label="Φ [V]")
        plt.tight_layout()
        if save_dir_path is not None:
            fig2.savefig(save_dir_path / "phi.png", dpi=300)

        if show:
            plt.show()
        else:
            # Close figures to prevent blocking in batch mode
            plt.close(fig1)
            plt.close(fig2)

    # ---------------------------------------------------------------------
    # Persistence helpers
    # ---------------------------------------------------------------------
    def save_results(self, output_dir: Path | str = "results") -> None:
        """Save simulation data (arrays + metadata) to `output_dir`.

        Saves:
        • results.npz … np.savez_compressed with nu_opt, nu_smoothed, Phi, x, y
        • optimisation.txt … summary of optimisation result
        If `plot_results` has not been called with the same directory, figures
        will not be included automatically (call plot_results with save_dir).
        """
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before saving results.")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Save arrays
        np.savez_compressed(
            out_path / "results.npz",
            nu_opt=self.nu_opt,
            nu_smoothed=self.nu_smoothed,
            Phi=self.Phi,
            x=self.x,
            y=self.y,
        )

        # Save optimisation summary
        with open(out_path / "optimisation.txt", "w", encoding="utf-8") as f:
            f.write(str(self.optimisation_result))

        print(f"Results saved to {out_path.resolve()}") 
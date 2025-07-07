import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path

from scipy.optimize import basinhopping
from scipy.interpolate import griddata, interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

@dataclass
class SimulationConfig:
    e: float = 1.602e-19
    epsilon_0: float = 8.854e-12
    h: float = 6.62607015e-34
    B: float = 13.0
    dt: float = 30e-9
    db: float = 30e-9
    epsilon_perp: float = 3.0
    epsilon_parallel: float = 6.6
    margin: float = 0.0
    potential_file: str = "data/0-data/VNS=2.1.txt"
    potential_scale: float = 1.0
    potential_offset: float = 0.0
    exc_file: str = "data/0-data/Exc_data_digitized.csv"
    niter: int = 5
    lbfgs_maxiter: int = 1000
    lbfgs_maxfun: int = 100000
    Nx: int = 64
    Ny: int = 64
    n_potentials: int = 0
    potential_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

class ThomasFermiSolver2:
    """Thomas–Fermi solver for 2D electron gas in magnetic field (no FFT)."""

    J_to_meV = 1.0 / (1.602e-22)

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.Nx = cfg.Nx
        self.Ny = cfg.Ny
        self.n_potentials = cfg.n_potentials
        self._load_external_potential()
        self._prepare_grid()
        self._prepare_kernels()
        self._prepare_exc_table()
        self._init_density()

    def _load_external_potential(self):
        if self.cfg.potential_data is not None:
            x_nm, y_nm, V_vals = self.cfg.potential_data
            self.x_data = np.asarray(x_nm) * 1e-9
            self.y_data = np.asarray(y_nm) * 1e-9
            self.V_data = np.asarray(V_vals)
            return
        data = np.loadtxt(self.cfg.potential_file, comments="%")
        if data.shape[1] < 3:
            raise ValueError("Potential file must have ≥3 columns (x, y, V).")
        self.x_data = data[:, 0] * 1e-9
        self.y_data = data[:, 1] * 1e-9
        self.V_data = data[:, -1]

    def _prepare_grid(self):
        cfg = self.cfg
        x_min, x_max = self.x_data.min() - cfg.margin, self.x_data.max() + cfg.margin
        y_min, y_max = self.y_data.min() - cfg.margin, self.y_data.max() + cfg.margin
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.Lx, self.Ly = (x_max - x_min), (y_max - y_min)
        self.dx, self.dy = self.Lx / self.Nx, self.Ly / self.Ny
        self.dA = self.dx * self.dy
        self.A_total = self.Lx * self.Ly
        self.x = np.linspace(x_min, x_max, self.Nx)
        self.y = np.linspace(y_min, y_max, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")
        Phi_grid = griddata(
            points=np.column_stack((self.x_data, self.y_data)),
            values=self.V_data,
            xi=(self.X, self.Y),
            method="linear",
        )
        mask = np.isnan(Phi_grid)
        if np.any(mask):
            Phi_grid[mask] = griddata(
                points=np.column_stack((self.x_data, self.y_data)),
                values=self.V_data,
                xi=(self.X[mask], self.Y[mask]),
                method="nearest",
            )
        Phi_grid *= cfg.potential_scale
        Phi_grid += cfg.potential_offset
        self.Phi = Phi_grid

    def _prepare_kernels(self):
        cfg = self.cfg
        self.D = cfg.e * cfg.B / cfg.h
        self.ell_B = np.sqrt(cfg.h / (2 * np.pi * cfg.e * cfg.B))
        # Precompute real-space Coulomb kernel
        x = np.linspace(self.x_min, self.x_max, self.Nx)
        y = np.linspace(self.y_min, self.y_max, self.Ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        self.Coulomb_kernel = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            for j in range(self.Ny):
                dx = X - X[i, j]
                dy = Y - Y[i, j]
                r = np.sqrt(dx**2 + dy**2) + 1e-12
                # 2D Coulomb potential (screened)
                self.Coulomb_kernel[i, j] = np.sum(1.0 / r)
        self.Coulomb_kernel *= self.cfg.e**2 / (4 * np.pi * self.cfg.epsilon_0)

    def _prepare_exc_table(self):
        exc_data = np.loadtxt(self.cfg.exc_file, delimiter=",", skiprows=1)
        n_exc, Exc_vals = exc_data[:, 0], exc_data[:, 1]
        self.exc_interp = interp1d(n_exc, Exc_vals, kind="linear")

    def _init_density(self):
        nu0 = np.clip(0.5 - 1.0 * (self.Phi - np.median(self.Phi)), 0.0, 1.0)
        self.nu0 = nu0.flatten()

    def gaussian_convolve(self, arr: np.ndarray) -> np.ndarray:
        sigma = self.ell_B / self.dx
        return gaussian_filter(arr, sigma=sigma, mode="wrap")

    def energy(self, nu_flat: np.ndarray) -> float:
        nu = nu_flat.reshape((self.Nx, self.Ny))
        nu_eff = self.gaussian_convolve(nu)
        n_eff = nu_eff * self.D

        # Hartree (Coulomb) energy via direct double sum (no FFT)
        E_C = 0.0
        for i in range(self.Nx):
            for j in range(self.Ny):
                for k in range(self.Nx):
                    for l in range(self.Ny):
                        if (i, j) != (k, l):
                            E_C += n_eff[i, j] * n_eff[k, l] * self.Coulomb_kernel[i, j]
        E_C *= 0.5 * self.dA**2 * self.J_to_meV / self.A_total

        E_phi = np.sum(-self.cfg.e * self.Phi * n_eff) * self.dA * self.J_to_meV
        E_xc = np.sum(self.exc_interp(nu_eff)) * self.dA * self.D
        total = E_phi + E_xc + E_C
        return float(total)

    def optimise(self):
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

    def plot_results(self, save_dir: Path | str | None = None):
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before plotting results.")
        extent = (self.x_min * 1e9, self.x_max * 1e9, self.y_min * 1e9, self.y_max * 1e9)
        save_dir_path = Path(save_dir) if save_dir is not None else None
        if save_dir_path is not None:
            save_dir_path.mkdir(parents=True, exist_ok=True)
        fig1 = plt.figure(figsize=(6, 5))
        plt.imshow(self.nu_smoothed, extent=extent, origin="lower", cmap="inferno")
        plt.title("Optimised Filling Factor ν(r)")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.colorbar(label="ν")
        plt.tight_layout()
        if save_dir_path is not None:
            fig1.savefig(save_dir_path / "nu_smoothed.png", dpi=300)
        fig2 = plt.figure(figsize=(6, 5))
        plt.imshow(self.Phi, extent=extent, origin="lower", cmap="viridis")
        plt.title("External Potential Φ(r) [V]")
        plt.xlabel("x [nm]")
        plt.ylabel("y [nm]")
        plt.colorbar(label="Φ [V]")
        plt.tight_layout()
        if save_dir_path is not None:
            fig2.savefig(save_dir_path / "phi.png", dpi=300)
        plt.show()

    def save_results(self, output_dir: Path | str = "results") -> None:
        if not hasattr(self, "nu_smoothed"):
            raise RuntimeError("Run optimise() before saving results.")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path / "results.npz",
            nu_opt=self.nu_opt,
            nu_smoothed=self.nu_smoothed,
            Phi=self.Phi,
            x=self.x,
            y=self.y,
        )
        with open(out_path / "optimisation.txt", "w", encoding="utf-8") as f:
            f.write(str(self.optimisation_result))
        print(f"Results saved to {out_path.resolve()}") 
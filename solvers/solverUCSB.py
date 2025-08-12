"""Analytical external-potential solver (UCSB variant) based on solver2.

This solver computes the external potential Φ(r) analytically from a provided
top-gate voltage map V_t(r) and a scalar back-gate voltage V_B, without reading
any potential data from files.

Analytical relation used in Fourier space (interpreted for Φ in volts):

    Φ(q) = V_t(q) * sinh(β d_b |q|) / sinh(β (d_t + d_b) |q|)
    Φ_offset_real = - V_B * d_t / (d_t + d_b)   (spatially uniform, real space)

where β = sqrt(ε_parallel / ε_perp). The factor −e is applied later in the
energy term, so Φ here remains an electrostatic potential in volts.

Inputs:
- Either provide `Vt_grid` with shape (Nx, Ny) in volts on the chosen domain,
  or leave it as None for a zero top-gate (only back-gate offset is used).
- Domain bounds are set via x/y min/max in nanometers.

All other physics, kernels, energies, and optimisation routines are reused from
`solver2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from solvers.solver2 import (
    SimulationConfig as _BaseConfig,
    ThomasFermiSolver as _BaseSolver,
)


# -----------------------------------------------------------------------------
# 1) Configuration – adds domain bounds, back-gate voltage, and V_t map
# -----------------------------------------------------------------------------


@dataclass
class SimulationConfig(_BaseConfig):
    """UCSB configuration extending solver2's parameters.

    New parameters
    -------------
    V_B : float
        Back-gate voltage [V]. Contributes a spatially uniform shift.
    Vt_grid : Optional[np.ndarray]
        Top-gate voltage distribution on the simulation grid [V], shape (Nx, Ny).
        If None, assumed to be zeros.
    x_min_nm, x_max_nm, y_min_nm, y_max_nm : float
        Domain bounds in nanometers. Used to construct the grid directly (no
        potential data files are read).
    """

    V_B: float = 0.0
    Vt_grid: Optional[np.ndarray] = None

    # Domain bounds (nm) – default to a 300 nm × 300 nm square centered at 0
    x_min_nm: float = -150.0
    x_max_nm: float = 150.0
    y_min_nm: float = -150.0
    y_max_nm: float = 150.0

    # Identify this solver implementation
    solver_type: str = "solverUCSB"


# -----------------------------------------------------------------------------
# 2) Solver – overrides grid/potential preparation to use analytic Φ
# -----------------------------------------------------------------------------


class ThomasFermiSolver(_BaseSolver):
    """Thomas–Fermi solver using an analytic external potential Φ(r).

    Differences to the base solver:
    - Does not read external potential from files
    - Builds the grid from explicit domain bounds
    - Computes Φ via FFT from V_t and adds a uniform back-gate offset
    """

    def __init__(self, cfg: SimulationConfig):  # type: ignore[override]
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Override: skip file I/O – nothing to load for the potential
    # ------------------------------------------------------------------
    def _load_external_potential(self) -> None:  # type: ignore[override]
        # Intentionally do nothing – all inputs come from configuration
        # (_prepare_grid will construct the grid and compute Φ)
        return

    # ------------------------------------------------------------------
    # Override: construct grid from bounds and compute Φ analytically
    # ------------------------------------------------------------------
    def _prepare_grid(self) -> None:  # type: ignore[override]
        cfg: SimulationConfig = self.cfg  # type: ignore[assignment]

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
        epsilon_hBN_perp = cfg.epsilon_perp
        epsilon_hBN_parallel = cfg.epsilon_parallel
        beta = np.sqrt(epsilon_hBN_parallel / epsilon_hBN_perp)

        # Ratio R(q) = sinh(β d_b |q|) / sinh(β (d_t + d_b) |q|) with proper q→0 limit
        dt = cfg.dt
        db = cfg.db
        denom = np.sinh(beta * (dt + db) * q)
        numer = np.sinh(beta * db * q)

        Rq = np.empty_like(q)
        # Handle q=0 separately to avoid 0/0; limit is db/(dt+db)
        zero_mask = (q == 0)
        Rq[zero_mask] = db / (dt + db)
        nonzero = ~zero_mask
        Rq[nonzero] = numer[nonzero] / denom[nonzero]

        # FFT of Vt and apply the transfer function
        Vt_q = np.fft.fft2(Vt)
        Phi_q = Vt_q * Rq
        Phi_from_top = np.real(np.fft.ifft2(Phi_q))

        # Back-gate uniform offset in real space
        phi_offset = -cfg.V_B * dt / (dt + db)

        Phi_grid = Phi_from_top + phi_offset

        # Apply any additional scaling/offset from config (kept for consistency)
        Phi_grid = Phi_grid * cfg.potential_scale + cfg.potential_offset

        self.Phi = Phi_grid

    # The rest (kernels, energies, plotting, etc.) is inherited unchanged


# Convenience aliases (match repository style)
SimulationConfigUCSB = SimulationConfig
ThomasFermiSolverUCSB = ThomasFermiSolver



from __future__ import annotations

"""Random-initialised variant of solver2.

This module provides *RandomSimulationConfig* and *RandomThomasFermiSolver* which
are identical to solver2's classes except that the initial filling-factor field
ν₀ is **purely random** (uniform 0–1) instead of the heuristic based on the
external potential.  Optionally a fixed random seed can be supplied for
reproducibility.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from solvers.solver2 import (
    SimulationConfig as _BaseConfig,
    ThomasFermiSolver as _BaseSolver,
)

# -----------------------------------------------------------------------------
# 1. Configuration dataclass (inherits every field from solver2.SimulationConfig)
# -----------------------------------------------------------------------------


@dataclass
class RandomSimulationConfig(_BaseConfig):
    """Configuration with optional *random_seed* for the random initial ν₀."""

    random_seed: Optional[int] = 42  # seed passed to NumPy for reproducibility

    # Let IDEs / type checkers know which solver this belongs to
    solver_type: str = "solver2_random"


# -----------------------------------------------------------------------------
# 2. Solver with random initial density
# -----------------------------------------------------------------------------


class RandomThomasFermiSolver(_BaseSolver):
    """Thomas–Fermi solver that starts from a completely random ν field."""

    def __init__(self, cfg: RandomSimulationConfig):  # type: ignore[override]
        # Set global / local seed for reproducibility (does not touch random module)
        if cfg.random_seed is not None:
            np.random.seed(cfg.random_seed)
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Override only the initial density routine
    # ------------------------------------------------------------------

    def _init_density(self):  # type: ignore[override]
        # Completely random ν₀ ∈ [0, 1]
        self.nu0 = np.random.rand(self.Nx * self.Ny)


# Convenience aliases
SimulationConfigRandom = RandomSimulationConfig
ThomasFermiSolverRandom = RandomThomasFermiSolver 
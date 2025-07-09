### Works for solver3.py ###

import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import time

from solver3_movie import SimulationConfig, ThomasFermiSolver

# -----------------------------------------------------------------------------
# USER-CONFIGURABLE PARAMETERS
# -----------------------------------------------------------------------------

# List of desired (V_QPC, V_SG) pairs in volts that you wish to simulate.
# These must exist in the header of data/1-data/James.txt
DESIRED_PAIRS: list[tuple[float, float]] = [(0.20, -1.50)]

# Square grid size N (replaces Nx, Ny)
GRID_N: int = 32

# Optimiser parameters
BASINHOPPING_NITER: int = 10
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 200000

# Potential offset / scaling (empirical)
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.033  # V

# -----------------------------------------------------------------------------


def parse_header(james_path: Path) -> dict[int, tuple[float, float]]:
    """Return mapping idx → (V_QPC, V_SG) from metadata header."""
    header_lines: list[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    header_text = " ".join(header_lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    mapping: dict[int, tuple[float, float]] = {}
    for m in pat.finditer(header_text):
        mapping[int(m.group(1)) - 1] = (float(m.group(2)), float(m.group(3)))
    return mapping


def _run_single_simulation(idx: int,
                           x_nm: np.ndarray,
                           y_nm: np.ndarray,
                           V_vals: np.ndarray,
                           pair: tuple[float, float],
                           batch_dir_str: str) -> int:
    """Run one simulation for a single potential column and save outputs."""

    cfg = SimulationConfig(
        N=GRID_N,
        potential_data=(x_nm, y_nm, V_vals),
        niter=BASINHOPPING_NITER,
        lbfgs_maxiter=LBFGS_MAXITER,
        lbfgs_maxfun=LBFGS_MAXFUN,
        potential_scale=POTENTIAL_SCALE,
        potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_digitized.csv",
        solver_type="solver3",
    )

    solver = ThomasFermiSolver(cfg)
    pot_dir = Path(batch_dir_str) / f"pot{idx}"
    pot_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1) Save simulation parameters BEFORE running optimisation
    # -------------------------------------------------------------
    from dataclasses import asdict

    with (pot_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    # -------------------------------------------------------------
    # 2) Run optimisation & measure execution time
    # -------------------------------------------------------------
    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    # -------------------------------------------------------------
    # 3) Plot results & save arrays / metadata
    # -------------------------------------------------------------
    title_extra = f"V_QPC={pair[0]:+.2f} V, V_SG={pair[1]:+.2f} V"
    solver.plot_results(save_dir=str(pot_dir), title_extra=title_extra, show=False)

    solver.save_results(pot_dir)

    # Add execution time info to existing file
    with (pot_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")

    return idx


def main() -> None:
    # ---------------------- load combined potential file --------------------
    data = np.loadtxt("data/1-data/James.txt", comments="%")

    mask = (
        (data[:, 0] >= -150) & (data[:, 0] <= 150) &
        (data[:, 1] >= -150) & (data[:, 1] <= 150)
    )
    x_nm = data[mask, 0]
    y_nm = data[mask, 1]
    V_columns = data[mask, 3:]

    james_path = Path("data/1-data/James.txt")
    idx_to_vs = parse_header(james_path)

    # Map requested pairs → column indices
    idxs: list[int] = []
    tol = 1e-6
    for pair in DESIRED_PAIRS:
        found = [i for i, vs in idx_to_vs.items() if abs(vs[0]-pair[0]) < tol and abs(vs[1]-pair[1]) < tol]
        if not found:
            raise ValueError(f"Requested pair {pair} not found in James.txt header.")
        idxs.append(found[0])

    # ---------------------- prepare output directories ----------------------
    base_dir = Path("analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------- run simulations ---------------------------
    tasks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], str]] = []
    for idx in idxs:
        V_vals = V_columns[:, idx]
        pair = list(idx_to_vs.values())[idx]
        tasks.append((idx, x_nm, y_nm, V_vals, pair, str(batch_dir)))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_simulation, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                finished_idx = fut.result()
                print(f"Finished simulation for idx={finished_idx}")
            except Exception as e:
                print(f"Simulation failed: {e}")

    print(f"All simulations complete. Results stored in {batch_dir}")


if __name__ == "__main__":
    main() 
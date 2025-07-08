# Using solver2 implementation (FFT-based Gaussian convolution)
from solver2 import SimulationConfig, ThomasFermiSolver
from datetime import datetime
from pathlib import Path
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------
# Specify the potentials you want to simulate by physical gate voltages.
# Give one or more (V_QPC, V_SG) tuples in volts.  Example:
#     desired_pairs = [(-0.40, -1.35), (-0.40, -1.20)]
# ------------------------------------------------------------

desired_pairs = [(-4.00, -1.50),(-3.70, -1.50),(-3.40, -1.50),(-3.10, -1.50),(-2.80, -1.50),(-2.50, -1.50),(-2.20, -1.50),(-1.90, -1.50),(-1.60, -1.50),(-1.30, -1.50),(-1.00, -1.50),(-0.70, -1.50),(-0.40, -1.50),(-0.10, -1.50),(0.20, -1.50),(0.50, -1.50),(0.80, -1.50),(1.10, -1.50),(1.40, -1.50),(1.70, -1.50),(2.00, -1.50),(2.30, -1.50),(2.60, -1.50),(2.90, -1.50),(3.20, -1.50),(3.50, -1.50),(3.80, -1.50)]  # <-- EDIT THIS LIST


def parse_header(james_path: Path):
    """Return mapping idx → (VQPC, VSG) floats from the header of *James.txt*."""
    header_lines = []
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

# ------------------------------------------------------------
# Helper function that runs a single simulation (must be at top level
# so that it can be pickled by multiprocessing).
# ------------------------------------------------------------


def _run_single_simulation(idx: int,
                           x_nm: np.ndarray,
                           y_nm: np.ndarray,
                           V_vals: np.ndarray,
                           pair: tuple[float, float],
                           batch_dir_str: str) -> int:
    """Run one Thomas-Fermi simulation and save results.

    Parameters
    ----------
    idx : int
        Column index of the potential.
    x_nm, y_nm : np.ndarray
        Spatial coordinates (nanometres).
    V_vals : np.ndarray
        Potential values (volts).
    pair : (V_QPC, V_SG)
        Gate voltages for title / bookkeeping.
    batch_dir_str : str
        Directory where results for this batch are stored.

    Returns
    -------
    int
        The same idx (used only for progress reporting).
    """

    cfg = SimulationConfig(
        potential_data=(x_nm, y_nm, V_vals),
        B=13.0,
        niter=1,
        lbfgs_maxiter=1000,
        lbfgs_maxfun=1000000,
        Nx=128,
        Ny=128,
        n_potentials=1,  # single run
        potential_scale=1.0,
        potential_offset=0.033,
        solver_type="solver2",
        exc_file="data/0-data/Exc_data_digitized.csv",
    )

    solver = ThomasFermiSolver(cfg)
    pot_dir = Path(batch_dir_str) / f"pot{idx}"
    pot_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Write simulation_parameters.txt BEFORE running the simulation
    # -----------------------------------------------------------------
    from dataclasses import asdict
    cfg_dict = asdict(cfg)
    param_file = pot_dir / "simulation_parameters.txt"
    with param_file.open("w", encoding="utf-8") as f:
        for k, v in cfg_dict.items():
            f.write(f"{k} = {v}\n")

    # -----------------------------------------------------------------
    # 2) Run simulation and measure time
    # -----------------------------------------------------------------
    import time
    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    # -----------------------------------------------------------------
    # 3) Plot results and manually save arrays / optimisation summary
    #    (avoid overwriting simulation_parameters.txt)
    # -----------------------------------------------------------------
    title_extra = f"V_QPC={pair[0]:+.2f} V, V_SG={pair[1]:+.2f} V"
    solver.plot_results(save_dir=str(pot_dir), title_extra=title_extra)

    # Save arrays
    np.savez_compressed(
        pot_dir / "results.npz",
        nu_opt=solver.nu_opt,
        nu_smoothed=solver.nu_smoothed,
        Phi=solver.Phi,
        x=solver.x,
        y=solver.y,
    )

    # Save optimisation summary
    with (pot_dir / "optimisation.txt").open("w", encoding="utf-8") as f:
        f.write(str(solver.optimisation_result))

    # Write execution time file
    with (pot_dir / "execution_time.txt").open("w", encoding="utf-8") as f:
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")

    return idx

def main():
    # Create configuration (adjust filenames / parameters as needed)

    # Extract spatial coordinates (nm) and potential columns
    data = np.loadtxt("data/1-data/Symmetry.txt", comments="%")
    x_nm = data[:, 0]
    y_nm = data[:, 1]
    # z_nm = data[:, 2]  # ignored for 2D simulation
    V_columns = data[:, 3:]

    # Map requested (V_QPC, V_SG) pairs to column indices ------------------
    james_path = Path("data/1-data/Symmetry.txt")
    idx_to_vs = parse_header(james_path)

    # tolerance for float comparison (absolutes in volts)
    tol = 1e-6
    idxs: list[int] = []
    for pair in desired_pairs:
        found = [i for i, vs in idx_to_vs.items() if abs(vs[0] - pair[0]) < tol and abs(vs[1] - pair[1]) < tol]
        if not found:
            raise ValueError(f"Requested pair {pair} not found in James.txt header.")
        idxs.append(found[0])

    # ---------------------------------------------------------------------

    # Prepare date-based parent folder only once
    base_dir = Path("analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    # Move existing timestamped runs for today into date_dir (once per script run)
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{today}_") and p.parent != date_dir:
            p.rename(date_dir / p.name)

    # Create new timestamped parent directory for *this* batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Prepare argument list for parallel execution
    tasks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[float, float], str]] = []
    for idx in idxs:
        V_vals = V_columns[:, idx]
        pair = list(idx_to_vs.values())[idx]
        tasks.append((idx, x_nm, y_nm, V_vals, pair, str(batch_dir)))

    # Run simulations in parallel (use all available CPUs by default)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_simulation, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                finished_idx = fut.result()
                print(f"Finished simulation for idx={finished_idx}")
            except Exception as e:
                print(f"Simulation failed: {e}")


if __name__ == "__main__":
    main() 
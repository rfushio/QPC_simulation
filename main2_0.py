from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from solvers.solver4 import SimulationConfig, ThomasFermiSolver
import re


# -----------------------------------------------------------------------------
# USER CONFIGURATION (mirrors mainUCSB, except external Φ comes from data)
# -----------------------------------------------------------------------------

WINDOWS: bool = False  # select output root (Windows vs default)

# Grid resolution (square)
GRID_N: int = 64

# Optimisation parameters
BASINHOPPING_NITER: int = 100
BASINHOPPING_STEP_SIZE: float = 0.1
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000

# Magnetic field (kept consistent with UCSB solver)
B_FIELD_T: float = 13.0

# Material / geometry
D_T: float = 30.0
D_B: float = 30.0

# Potential scaling
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0

# Exchange–correlation scaling – allow multiple, run combinations
XC_SCALES: list[float] = [1.8,1.51]

# Progressive refinement
MATRYOSHKA: bool = True
COARSE_ACCEPT_LIMIT: int = 3

# Parallel execution toggle
PARALLEL: bool = True
import os
MAX_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)

# Mode selection: "files" (list of separate potential files) or "combined"
POTENTIAL_MODE: str = "combined"  # or "files"

# Mode: files — External potential files; each provides x_nm, y_nm, Phi[V]
POTENTIAL_FILES: list[str] = []

# Mode: combined — One combined file with many columns and header mapping
COMBINED_FILE: str = "data/1-data/James2.txt"  # x_nm, y_nm, columns of Phi
# Specify desired pairs like main1_5: list of (V_QPC, V_SG)
DESIRED_PAIRS: list[tuple[float, float]] = [(-1.90,-1.50),(-1.60,-1.50),(-1.30,-1.50),(-1.00,-1.50),(-0.70,-1.50)]


def run_single_file(potential_file: str, xc_scale: float, out_dir: Path) -> None:
    cfg = SimulationConfig(
        Nx=GRID_N,
        Ny=GRID_N,
        B=B_FIELD_T,
        dt=D_T * 1e-9,
        db=D_B * 1e-9,
        potential_file=potential_file,
        potential_scale=POTENTIAL_SCALE,
        potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new.csv",
        solver_type="solver4",
        exc_scale=float(xc_scale),
        use_matryoshka=MATRYOSHKA,
        niter=BASINHOPPING_NITER,
        step_size=BASINHOPPING_STEP_SIZE,
        lbfgs_maxiter=LBFGS_MAXITER,
        lbfgs_maxfun=LBFGS_MAXFUN,
        coarse_accept_limit=COARSE_ACCEPT_LIMIT,
    )

    solver = ThomasFermiSolver(cfg)

    # Save parameters pre-run
    with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    title_extra = f"file={Path(potential_file).name}, XC={float(xc_scale):.3f}"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")

def run_single_data(x_nm: np.ndarray, y_nm: np.ndarray, V_vals: np.ndarray, xc_scale: float, title_tag: str, out_dir: Path) -> None:
    # Write a temporary potential file (x_nm, y_nm, Phi[V]) for this case
    tmp_pot_file = out_dir / "external_potential.txt"
    arr = np.column_stack([np.asarray(x_nm), np.asarray(y_nm), np.asarray(V_vals)])
    np.savetxt(tmp_pot_file, arr, fmt="%.9f", header="x_nm y_nm Phi_V", comments="% ")

    cfg = SimulationConfig(
        Nx=GRID_N,
        Ny=GRID_N,
        B=B_FIELD_T,
        dt=D_T * 1e-9,
        db=D_B * 1e-9,
        potential_file=str(tmp_pot_file),
        potential_scale=POTENTIAL_SCALE,
        potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new.csv",
        solver_type="solver4",
        exc_scale=float(xc_scale),
        use_matryoshka=MATRYOSHKA,
        niter=BASINHOPPING_NITER,
        step_size=BASINHOPPING_STEP_SIZE,
        lbfgs_maxiter=LBFGS_MAXITER,
        lbfgs_maxfun=LBFGS_MAXFUN,
        coarse_accept_limit=COARSE_ACCEPT_LIMIT,
    )

    solver = ThomasFermiSolver(cfg)

    with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    title_extra = f"{title_tag}, XC={float(xc_scale):.3f}"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")


def _parse_header_pairs(james_path: Path) -> dict[int, tuple[float, float]]:
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


def main() -> None:
    # Output root
    base_dir = Path("analysis_folder_windows" if WINDOWS else "analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / (timestamp + "_MAIN2_0")
    batch_dir.mkdir(parents=True, exist_ok=True)

    if POTENTIAL_MODE == "files":
        if not PARALLEL:
            idx = 0
            for pot_file in POTENTIAL_FILES:
                for xc in XC_SCALES:
                    out = batch_dir / f"case_{idx:02d}"
                    out.mkdir(parents=True, exist_ok=True)
                    try:
                        run_single_file(pot_file, xc, out)
                        print(f"Finished case {idx}: {Path(pot_file).name}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {idx} failed: {e}")
                    idx += 1
        else:
            from concurrent.futures import ProcessPoolExecutor
            futures = []
            idx = 0
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for pot_file in POTENTIAL_FILES:
                    for xc in XC_SCALES:
                        out = batch_dir / f"case_{idx:02d}"
                        out.mkdir(parents=True, exist_ok=True)
                        futures.append((idx, pot_file, xc, ex.submit(run_single_file, pot_file, xc, out)))
                        idx += 1
                for i, pot_file, xc, fut in futures:
                    try:
                        fut.result()
                        print(f"Finished case {i}: {Path(pot_file).name}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {i} failed: {e}")
    elif POTENTIAL_MODE == "combined":
        # Load combined potential data and map pairs to columns
        data = np.loadtxt(COMBINED_FILE, comments="%")
        mask = (
            (data[:, 0] >= -150) & (data[:, 0] <= 150) &
            (data[:, 1] >= -150) & (data[:, 1] <= 150)
        )
        x_nm = data[mask, 0]
        y_nm = data[mask, 1]
        V_columns = data[mask, 3:]

        idx_to_vs = _parse_header_pairs(Path(COMBINED_FILE))
        # Build list of indices for desired pairs
        idxs: list[int] = []
        tol = 1e-6
        for pair in DESIRED_PAIRS:
            found = [i for i, vs in idx_to_vs.items() if abs(vs[0]-pair[0]) < tol and abs(vs[1]-pair[1]) < tol]
            if not found:
                raise ValueError(f"Requested pair {pair} not found in header of {COMBINED_FILE}")
            idxs.append(found[0])

        if not PARALLEL:
            idx_case = 0
            for col_idx in idxs:
                V_vals = V_columns[:, col_idx]
                pair = list(idx_to_vs.values())[col_idx]
                tag = f"V_QPC={pair[0]:+.2f} V, V_SG={pair[1]:+.2f} V"
                for xc in XC_SCALES:
                    out = batch_dir / f"case_{idx_case:02d}"
                    out.mkdir(parents=True, exist_ok=True)
                    try:
                        run_single_data(x_nm, y_nm, V_vals, xc, tag, out)
                        print(f"Finished case {idx_case}: {tag}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {idx_case} failed: {e}")
                    idx_case += 1
        else:
            from concurrent.futures import ProcessPoolExecutor
            futures = []
            idx_case = 0
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for col_idx in idxs:
                    V_vals = V_columns[:, col_idx]
                    pair = list(idx_to_vs.values())[col_idx]
                    tag = f"V_QPC={pair[0]:+.2f} V, V_SG={pair[1]:+.2f} V"
                    for xc in XC_SCALES:
                        out = batch_dir / f"case_{idx_case:02d}"
                        out.mkdir(parents=True, exist_ok=True)
                        futures.append((idx_case, tag, xc, ex.submit(run_single_data, x_nm, y_nm, V_vals, xc, tag, out)))
                        idx_case += 1
                for i, tag, xc, fut in futures:
                    try:
                        fut.result()
                        print(f"Finished case {i}: {tag}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {i} failed: {e}")
    else:
        raise ValueError("POTENTIAL_MODE must be 'files' or 'combined'")

    print(f"All MAIN2_0 simulations complete. Results stored in {batch_dir}")


if __name__ == "__main__":
    main()



import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import argparse
import os
import json
import csv

import numpy as np

from solvers.solverUCSB import SimulationConfig, ThomasFermiSolver
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# BASE CONFIGURATION (Copied from mainUCSB.py)
# -----------------------------------------------------------------------------

GRID_N: int = 128
BASINHOPPING_NITER: int = 10
BASINHOPPING_STEP_SIZE: float = 0.5
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000
B_FIELD_T: float = 13.0
BAR_WIDTH_NM: float = 70
D_T: float = 30.0 
D_B: float = 30.0
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0
XC_SCALE: float = 1.51
MATRYOSHKA: bool = False

def magnetic_length_m(B_T: float) -> float:
    h = 6.62607015e-34
    e = 1.602e-19
    return float(np.sqrt(h / (2.0 * np.pi * e * B_T)))

L_B_M: float = 7.11e-9
HALF_SPAN_M: float = 20.0 * L_B_M
X_MIN_NM: float = -HALF_SPAN_M * 1e9
X_MAX_NM: float = +HALF_SPAN_M * 1e9
Y_MIN_NM: float = -HALF_SPAN_M * 1e9
Y_MAX_NM: float = +HALF_SPAN_M * 1e9

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (Updated to dot version if needed)
# -----------------------------------------------------------------------------

def build_vt_grid(N: int, x_min_nm: float, x_max_nm: float, y_min_nm: float, y_max_nm: float, dot_radius_nm: float, V_in: float, V_out: float) -> np.ndarray:
    x_nm = np.linspace(x_min_nm, x_max_nm, N)
    y_nm = np.linspace(y_min_nm, y_max_nm, N)
    X_nm, Y_nm = np.meshgrid(x_nm, y_nm, indexing="ij")
    R_nm = np.sqrt(X_nm**2 + Y_nm**2)
    Vt = np.full((N, N), V_out, dtype=float)
    Vt[R_nm <= dot_radius_nm] = V_in
    return Vt

def run_single(V_in: float, V_out: float, V_B_case: float, out_dir: Path) -> None:
    vt_grid = build_vt_grid(GRID_N, X_MIN_NM, X_MAX_NM, Y_MIN_NM, Y_MAX_NM, BAR_WIDTH_NM, V_in, V_out)

    # Note: Vt_grid plotting is disabled for cluster jobs to save time/resources
    
    cfg = SimulationConfig(
        Nx=GRID_N, Ny=GRID_N, B=B_FIELD_T, V_B=float(V_B_case),
        dt=D_T*1e-9, db=D_B*1e-9, Vt_grid=vt_grid,
        v_in=float(V_in), v_out=float(V_out),
        x_min_nm=X_MIN_NM, x_max_nm=X_MAX_NM, y_min_nm=Y_MIN_NM, y_max_nm=Y_MAX_NM,
        niter=BASINHOPPING_NITER, step_size=BASINHOPPING_STEP_SIZE,
        lbfgs_maxiter=LBFGS_MAXITER, lbfgs_maxfun=LBFGS_MAXFUN,
        potential_scale=POTENTIAL_SCALE, potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new.csv", solver_type="solverUCSB",
        exc_scale=XC_SCALE, use_matryoshka=MATRYOSHKA,
    )

    solver = ThomasFermiSolver(cfg)
    solver.optimise()

    title_extra = f"V_in={V_in:+.3f} V, V_out={V_out:+.3f} V, V_B={float(V_B_case):+.3f} V"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

# -----------------------------------------------------------------------------
# CLUSTER EXECUTION MAIN BLOCK
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run UCSB Thomas-Fermi simulation on HPC or query job count.")
    parser.add_argument("--task-id", type=int, default=None, help="SLURM array task ID (e.g., 0-99).")
    parser.add_argument("--batch-folder", type=str, default=None, help="The parent folder for this batch of simulations.")
    parser.add_argument("--get-count", action="store_true", help="Print total number of parameter combinations and exit.")
    # Manual combo specification (either file or JSON string). If both are given, file takes precedence.
    parser.add_argument("--combos-file", type=str, default=None, help="Path to CSV or JSON file with explicit (V_NS,V_EW,V_B) tuples.")
    parser.add_argument("--combos-json", type=str, default=None, help="JSON string containing a list of [V_NS, V_EW, V_B] tuples.")
    args = parser.parse_args()

    # --- Define Parameter Space (explicit manual combos only) ---
    # The previous env-based grid specification is removed per user request.
    # Accepted inputs:
    #   - --combos-file PATH   (CSV with rows: V_NS,V_EW,V_B or JSON list of [V_NS,V_EW,V_B])
    #   - --combos-json '[[VNS,VEW,VB], ...]'

    def _read_combos_from_file(path: str) -> list[tuple[float, float, float]]:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Combos file not found: {path}")
        suffix = p.suffix.lower()
        combos: list[tuple[float, float, float]] = []
        if suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                raise SystemExit(f"Failed to parse JSON combos file: {e}")
            if not isinstance(data, (list, tuple)):
                raise SystemExit("Combos JSON must be a list of triples")
            for item in data:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    raise SystemExit("Each combo must be a list/tuple of length 3: [V_NS, V_EW, V_B]")
                vns, vew, vb = map(float, item)
                combos.append((vns, vew, vb))
            return combos
        # CSV or other text: parse as CSV
        try:
            with p.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row or all((c.strip() == "" for c in row)):
                        continue
                    # Skip header if non-numeric
                    try:
                        vns, vew, vb = map(float, row[:3])
                    except ValueError:
                        # probably header; skip this row and continue
                        continue
                    combos.append((vns, vew, vb))
        except Exception as e:
            raise SystemExit(f"Failed to read CSV combos file: {e}")
        if not combos:
            raise SystemExit("No valid combos found in the file.")
        return combos

    def _read_combos_from_json_str(json_str: str) -> list[tuple[float, float, float]]:
        try:
            data = json.loads(json_str)
        except Exception as e:
            raise SystemExit(f"Invalid --combos-json: {e}")
        if not isinstance(data, (list, tuple)):
            raise SystemExit("--combos-json must be a list of [V_NS, V_EW, V_B] triples")
        combos: list[tuple[float, float, float]] = []
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise SystemExit("Each combo must be a list/tuple of length 3: [V_NS, V_EW, V_B]")
            vns, vew, vb = map(float, item)
            combos.append((vns, vew, vb))
        return combos

    combos: list[tuple[float, float, float]]
    if args.combos_file:
        combos = _read_combos_from_file(args.combos_file)
    elif args.combos_json:
        combos = _read_combos_from_json_str(args.combos_json)
    else:
        # Also allow environment variables for batch submission convenience
        combos_file_env = os.getenv("COMBOS_FILE")
        combos_json_env = os.getenv("COMBOS_JSON")
        if combos_file_env:
            combos = _read_combos_from_file(combos_file_env)
        elif combos_json_env:
            combos = _read_combos_from_json_str(combos_json_env)
        else:
            raise SystemExit("You must provide combos via --combos-file, --combos-json, COMBOS_FILE, or COMBOS_JSON.")

    total_count = int(len(combos))

    # If only the count is requested, print and exit
    if args.get_count:
        print(total_count)
        return

    if args.task_id is None or args.batch_folder is None:
        raise SystemExit("--task-id and --batch-folder are required unless --get-count is specified.")

    task_id = int(args.task_id)
    if not 0 <= task_id < total_count:
        raise ValueError(f"Task ID {task_id} is out of the valid range 0..{total_count-1}.")

    # --- Map Task ID to Parameters ---
    try:
        vns, vew, vb = combos[task_id]
    except Exception:
        raise SystemExit(f"Invalid task-id {task_id} for combos list of length {total_count}")
    vns = float(vns); vew = float(vew); vb = float(vb)

    # --- Set up Output Directory ---
    case_dir = Path(args.batch_folder) / f"case_{task_id:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting Task ID: {task_id} ---")
    print(f"Parameters: V_NS={vns:+.3f}, V_EW={vew:+.3f}, V_B={vb:+.3f}")
    print(f"Output directory: {case_dir}")

    run_single(vns, vew, vb, case_dir)

    print(f"--- Finished Task ID: {task_id} ---")

if __name__ == "__main__":
    main()



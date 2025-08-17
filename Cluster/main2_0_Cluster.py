from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

# Lazy imports of numpy and matplotlib happen inside functions to allow --get-count without env

from solvers.solver4 import SimulationConfig, ThomasFermiSolver


# -----------------------------
# User-config defaults (match main2_0)
# -----------------------------
GRID_N: int = 64
B_FIELD_T: float = 13.0
D_T: float = 30.0
D_B: float = 30.0
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0
MATRYOSHKA: bool = False
BASINHOPPING_NITER: int = 1
BASINHOPPING_STEP_SIZE: float = 0.1
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000


def _parse_pairs_json(s: str) -> list[tuple[float, float]]:
    try:
        data = json.loads(s)
    except Exception as e:
        raise SystemExit(f"Invalid DESIRED_PAIRS_JSON: {e}")
    out: list[tuple[float, float]] = []
    if not isinstance(data, (list, tuple)):
        raise SystemExit("DESIRED_PAIRS_JSON must be a list of [V_QPC, V_SG]")
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SystemExit("Each desired pair must be [V_QPC, V_SG]")
        vq, vs = item
        out.append((float(vq), float(vs)))
    return out


def _parse_list_json(s: str, name: str) -> list:
    try:
        data = json.loads(s)
    except Exception as e:
        raise SystemExit(f"Invalid {name}: {e}")
    if not isinstance(data, (list, tuple)):
        raise SystemExit(f"{name} must be a JSON list")
    return list(data)


def _build_tasks_from_env() -> list[dict]:
    """Return list of task dicts.

    Env vars:
      POTENTIAL_MODE: 'files' | 'combined'
      POTENTIAL_FILES_JSON: ["path1", "path2", ...]
      COMBINED_FILE: path
      DESIRED_PAIRS_JSON: [[V_QPC, V_SG], ...]
      XC_SCALES_JSON: [1.8, 2.0, ...]
    """
    mode = os.getenv("POTENTIAL_MODE", "files").strip().lower()
    xc_scales = os.getenv("XC_SCALES_JSON")
    if xc_scales is None:
        xc_values = [1.8]
    else:
        xc_values = [float(x) for x in _parse_list_json(xc_scales, "XC_SCALES_JSON")]

    tasks: list[dict] = []
    if mode == "files":
        files_json = os.getenv("POTENTIAL_FILES_JSON", "[]")
        files = [str(x) for x in _parse_list_json(files_json, "POTENTIAL_FILES_JSON")]
        for f in files:
            for xc in xc_values:
                tasks.append({"kind": "file", "file": f, "xc": float(xc)})
    elif mode == "combined":
        combined = os.getenv("COMBINED_FILE")
        pairs_json = os.getenv("DESIRED_PAIRS_JSON", "[]")
        if not combined:
            raise SystemExit("COMBINED_FILE must be set for POTENTIAL_MODE=combined")
        pairs = _parse_pairs_json(pairs_json)
        for (vq, vs) in pairs:
            for xc in xc_values:
                tasks.append({"kind": "combined", "combined": combined, "pair": (float(vq), float(vs)), "xc": float(xc)})
    else:
        raise SystemExit("POTENTIAL_MODE must be 'files' or 'combined'")
    return tasks


def _run_file_task(task: dict, batch_folder: Path, task_id: int) -> None:
    import numpy as np  # noqa: F401  (used by solver)
    potential_file: str = task["file"]
    xc_scale: float = float(task["xc"])
    out_dir = batch_folder / f"case_{task_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SimulationConfig(
        Nx=GRID_N, Ny=GRID_N, B=B_FIELD_T,
        dt=D_T * 1e-9, db=D_B * 1e-9,
        potential_file=potential_file,
        potential_scale=POTENTIAL_SCALE, potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new2.csv", solver_type="solver4",
        exc_scale=xc_scale, use_matryoshka=MATRYOSHKA,
        lbfgs_maxiter=LBFGS_MAXITER, lbfgs_maxfun=LBFGS_MAXFUN,
        niter=BASINHOPPING_NITER, step_size=BASINHOPPING_STEP_SIZE,
    )
    solver = ThomasFermiSolver(cfg)

    with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    title_extra = f"file={Path(potential_file).name}, XC={xc_scale:.3f}"
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


def _run_combined_task(task: dict, batch_folder: Path, task_id: int) -> None:
    import numpy as np
    from pathlib import Path as _Path

    combined_file: str = task["combined"]
    vq, vs = task["pair"]
    xc_scale: float = float(task["xc"])

    data = np.loadtxt(combined_file, comments="%")
    mask = (
        (data[:, 0] >= -150) & (data[:, 0] <= 150) &
        (data[:, 1] >= -150) & (data[:, 1] <= 150)
    )
    x_nm = data[mask, 0]
    y_nm = data[mask, 1]
    V_columns = data[mask, 3:]

    idx_to_vs = _parse_header_pairs(_Path(combined_file))
    tol = 1e-6
    found = [i for i, vs_pair in idx_to_vs.items() if abs(vs_pair[0]-vq) < tol and abs(vs_pair[1]-vs) < tol]
    if not found:
        raise SystemExit(f"Requested pair ({vq},{vs}) not found in header of {combined_file}")
    col_idx = found[0]
    V_vals = V_columns[:, col_idx]

    out_dir = batch_folder / f"case_{task_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary potential file for this case
    tmp_pot_file = out_dir / "external_potential.txt"
    arr = np.column_stack([np.asarray(x_nm), np.asarray(y_nm), np.asarray(V_vals)])
    np.savetxt(tmp_pot_file, arr, fmt="%.9f", header="x_nm y_nm Phi_V", comments="% ")

    cfg = SimulationConfig(
        Nx=GRID_N, Ny=GRID_N, B=B_FIELD_T,
        dt=D_T * 1e-9, db=D_B * 1e-9,
        potential_file=str(tmp_pot_file),
        potential_scale=POTENTIAL_SCALE, potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new2.csv", solver_type="solver4",
        exc_scale=xc_scale, use_matryoshka=MATRYOSHKA,
        lbfgs_maxiter=LBFGS_MAXITER, lbfgs_maxfun=LBFGS_MAXFUN,
        niter=BASINHOPPING_NITER, step_size=BASINHOPPING_STEP_SIZE,
    )
    solver = ThomasFermiSolver(cfg)

    with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    title_extra = f"V_QPC={vq:+.2f} V, V_SG={vs:+.2f} V, XC={xc_scale:.3f}"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run main2_0 tasks on HPC or query job count.")
    parser.add_argument("--task-id", type=int, default=None, help="SLURM array task ID")
    parser.add_argument("--batch-folder", type=str, default=None, help="Parent folder for this batch")
    parser.add_argument("--get-count", action="store_true", help="Print total number of tasks and exit")
    args = parser.parse_args()

    tasks = _build_tasks_from_env()
    total = len(tasks)

    if args.get_count:
        print(total)
        return

    if args.task_id is None or args.batch_folder is None:
        raise SystemExit("--task-id and --batch-folder are required unless --get-count is specified.")

    if not (0 <= int(args.task_id) < total):
        raise SystemExit(f"task-id out of range 0..{total-1}")

    batch_folder = Path(args.batch_folder)
    batch_folder.mkdir(parents=True, exist_ok=True)

    task = tasks[int(args.task_id)]
    kind = task.get("kind")
    if kind == "file":
        _run_file_task(task, batch_folder, int(args.task_id))
    elif kind == "combined":
        _run_combined_task(task, batch_folder, int(args.task_id))
    else:
        raise SystemExit(f"Unknown task kind: {kind}")


if __name__ == "__main__":
    main()



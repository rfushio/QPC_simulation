from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, e, m_e

from solvers.solver4 import SimulationConfig, ThomasFermiSolver
import re


# -----------------------------------------------------------------------------
# USER CONFIGURATION (mirrors mainUCSB, except external Φ comes from data)
# -----------------------------------------------------------------------------

WINDOWS: bool = False  # select output root (Windows vs default)

# Grid resolution (square)
GRID_N: int = 32

# Simulation area setup
SIMULATION_AREA_MODE: str = "manual_nm"  # "auto_lB" or "manual_nm"
# Mode: auto_lB — Set simulation area as a multiple of the magnetic length l_B
SIMULATION_AREA_LB_MULTIPLE: float = 20.0
# Mode: manual_nm — Set simulation area manually in nanometers
SIMULATION_AREA_MANUAL_NM: tuple[float, float, float, float] = (-150.0, 150.0, -150.0, 150.0)

# Optimisation parameters
BASINHOPPING_NITER: int = 1
BASINHOPPING_STEP_SIZE: float = 0.5
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000

# Magnetic field (kept consistent with UCSB solver)
B_FIELD_T: float = 10.0

# Expected bulk filling factor ν (e.g., 0, 1, ...)
EXPECTED_BULK_NU: float = 1.0

# Material / geometry
D_T: float = 30.0
D_B: float = 30.0

# Potential scaling
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0

# Exchange–correlation scaling – allow multiple, run combinations
XC_SCALES: list[float] = [1.5]

# Progressive refinement
MATRYOSHKA: bool = True
COARSE_ACCEPT_LIMIT: int = 3

# Parallel execution toggle
PARALLEL: bool = True
import os
MAX_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)

# Correlation analysis toggle
CORRELATION_ANALYSIS: bool = True

# Mode selection: "files" (list of separate potential files) or "combined"
POTENTIAL_MODE: str = "combined"  # or "files"

# Mode: files — External potential files; each provides x_nm, y_nm, Phi[V]
POTENTIAL_FILES: list[str] = []

# Mode: combined — One combined file with many columns and header mapping
COMBINED_FILE: str = "data/2-data/dot-bridge-2.txt"  # x_nm, y_nm, columns of Phi
# Specify desired triplets: list of (V_TG, V_QPC, V_BG)
DESIRED_TRIPLETS: list[tuple[float, float, float]] = [
    #(0.07, -1.5, 0.07)
]
# If the combined file contains only a single varying parameter (e.g., V_QPC only),
# specify the desired V_QPC values here (optional). If empty, all columns are used.
DESIRED_VQPC_VALUES: list[float] = [-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4]


def run_single_file(potential_file: str, xc_scale: float, out_dir: Path) -> tuple[float, float]:
    cfg = SimulationConfig(
        Nx=GRID_N,
        Ny=GRID_N,
        B=B_FIELD_T,
        dt=D_T * 1e-9,
        db=D_B * 1e-9,
        potential_file=potential_file,
        potential_scale=POTENTIAL_SCALE,
        potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new2.csv",
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
        f.write(f"EXPECTED_BULK_NU = {EXPECTED_BULK_NU}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    # Electron counts
    expected_N = float(EXPECTED_BULK_NU) * float(solver.A_total) * float(solver.D)
    actual_N = float(np.sum(solver.nu_smoothed) * solver.dA * solver.D)
    delta_N = actual_N - expected_N

    title_extra = f"file={Path(potential_file).name}, XC={float(xc_scale):.3f}, ΔN={delta_N:+.2f}"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")
        f.write("\n# Electron count summary\n")
        f.write(f"expected_N_electrons = {expected_N:.6f}\n")
        f.write(f"actual_N_electrons = {actual_N:.6f}\n")
        f.write(f"delta_N_electrons = {delta_N:.6f}\n")

    # For files-mode, we cannot infer V_QPC; return NaN with ΔN
    return float("nan"), float(delta_N)

def run_single_data(x_nm: np.ndarray, y_nm: np.ndarray, V_vals: np.ndarray, xc_scale: float, title_tag: str, out_dir: Path) -> tuple[float, float]:
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
        exc_file="data/0-data/Exc_data_new2.csv",
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
        f.write(f"EXPECTED_BULK_NU = {EXPECTED_BULK_NU}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    # Electron counts
    expected_N = float(EXPECTED_BULK_NU) * float(solver.A_total) * float(solver.D)
    actual_N = float(np.sum(solver.nu_smoothed) * solver.dA * solver.D)
    delta_N = actual_N - expected_N

    title_extra = f"{title_tag}, XC={float(xc_scale):.3f}, ΔN={delta_N:+.2f}"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")
        f.write("\n# Electron count summary\n")
        f.write(f"expected_N_electrons = {expected_N:.6f}\n")
        f.write(f"actual_N_electrons = {actual_N:.6f}\n")
        f.write(f"delta_N_electrons = {delta_N:.6f}\n")

    # Try to parse V_QPC from title_tag 'VTG=..., V_QPC=..., V_BG=...'
    import re as _re
    m = _re.search(r"V_QPC=([+-]?\d*\.?\d+)", title_tag)
    v_qpc = float(m.group(1)) if m else float("nan")
    return float(v_qpc), float(delta_N)


def _parse_header_triplets(path: Path) -> dict[int, tuple[float, float, float]]:
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    header_text = " ".join(header_lines)
    pat = re.compile(
        r"@\s*(\d+):\s*VTG=([+-]?\d*\.?\d+)\s*V,\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VBG=([+-]?\d*\.?\d+)\s*V",
        re.I
    )
    mapping: dict[int, tuple[float, float, float]] = {}
    for m in pat.finditer(header_text):
        mapping[int(m.group(1)) - 1] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return mapping


def _parse_header_vqpc_only(path: Path) -> dict[int, float]:
    """Parse header mapping when only V_QPC was varied.

    Expected header tokens like:
        @ 1: V_QPC=+0.20 V
    or  @ 1: VQPC=+0.20 V
    Returns dict: column_index (0-based) -> V_QPC value [V]
    """
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    header_text = " ".join(header_lines)
    import re as _re
    # Match both 'V (V) @ VQPC=-1.5' style and '@ 1: V_QPC=-1.5 V' style
    # 1) Enumerate occurrences of VQPC=... or V_QPC=... in order
    pat_all = _re.compile(r"(?:@\s*)?(?:V_QPC|VQPC)\s*=\s*([+-]?\d*\.?\d+)", _re.I)
    matches = list(pat_all.finditer(header_text))
    mapping: dict[int, float] = {}
    for idx, m in enumerate(matches):
        try:
            mapping[idx] = float(m.group(1))
        except Exception:
            continue
    return mapping


def main() -> None:
    # Output root
    base_dir = Path("analysis_dot_folder_windows" if WINDOWS else "analysis_dot_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / (timestamp + "_MAIN2_DOT")
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Collect (x, y) for correlation plot: x = V_QPC (combined mode), y = ΔN
    correlation_pairs: list[tuple[float, float]] = []

    if POTENTIAL_MODE == "files":
        if not PARALLEL:
            idx = 0
            for pot_file in POTENTIAL_FILES:
                for xc in XC_SCALES:
                    out = batch_dir / f"case_{idx:02d}"
                    out.mkdir(parents=True, exist_ok=True)
                    try:
                        x_val, dN = run_single_file(pot_file, xc, out)
                        if CORRELATION_ANALYSIS:
                            correlation_pairs.append((x_val, dN))
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
                        x_val, dN = fut.result()
                        if CORRELATION_ANALYSIS:
                            correlation_pairs.append((float(x_val), float(dN)))
                        print(f"Finished case {i}: {Path(pot_file).name}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {i} failed: {e}")
    elif POTENTIAL_MODE == "combined":
        # Load combined potential data and map pairs to columns
        data = np.loadtxt(COMBINED_FILE, comments="%")

        if SIMULATION_AREA_MODE == "auto_lB":
            l_B = np.sqrt(hbar / (e * B_FIELD_T)) * 1e9  # Magnetic length in nm
            half_width = SIMULATION_AREA_LB_MULTIPLE * l_B / 2.0
            x_min, x_max = -half_width, half_width
            y_min, y_max = -half_width, half_width
        elif SIMULATION_AREA_MODE == "manual_nm":
            x_min, x_max, y_min, y_max = SIMULATION_AREA_MANUAL_NM
        else:
            raise ValueError("SIMULATION_AREA_MODE must be 'auto_lB' or 'manual_nm'")

        mask = (
            (data[:, 0] >= x_min) & (data[:, 0] <= x_max) &
            (data[:, 1] >= y_min) & (data[:, 1] <= y_max)
        )
        x_nm = data[mask, 0]
        y_nm = data[mask, 1]
        V_columns = data[mask, 3:]

        # Try mapping for triplets; if not found, fall back to V_QPC-only mapping
        try:
            idx_to_vs = _parse_header_triplets(Path(COMBINED_FILE))
            header_mode = "triplets"
        except Exception:
            idx_to_vs = {}
            header_mode = "none"

        idxs: list[int] = []
        tol = 1e-6
        if idx_to_vs:
            for triplet in DESIRED_TRIPLETS:
                found = [
                    i for i, vs in idx_to_vs.items()
                    if abs(vs[0] - triplet[0]) < tol
                    and abs(vs[1] - triplet[1]) < tol
                    and abs(vs[2] - triplet[2]) < tol
                ]
                if not found:
                    raise ValueError(f"Requested triplet {triplet} not found in header of {COMBINED_FILE}")
                idxs.append(found[0])
        else:
            # V_QPC-only path
            idx_to_vqpc = _parse_header_vqpc_only(Path(COMBINED_FILE))
            if DESIRED_VQPC_VALUES:
                for vqpc in DESIRED_VQPC_VALUES:
                    found = [i for i, v in idx_to_vqpc.items() if abs(v - vqpc) < tol]
                    if not found:
                        raise ValueError(f"Requested V_QPC={vqpc} not found in header of {COMBINED_FILE}")
                    idxs.append(found[0])
            else:
                # Use all available columns
                idxs = sorted(idx_to_vqpc.keys())

        if not PARALLEL:
            idx_case = 0
            for col_idx in idxs:
                V_vals = V_columns[:, col_idx]
                if idx_to_vs:
                    triplet = list(idx_to_vs.values())[col_idx]
                    tag = f"VTG={triplet[0]:+.2f} V, V_QPC={triplet[1]:+.2f} V, V_BG={triplet[2]:+.2f} V"
                else:
                    # V_QPC-only header; map by key to avoid order mismatch
                    idx_to_vqpc = _parse_header_vqpc_only(Path(COMBINED_FILE))
                    vqpc = idx_to_vqpc.get(col_idx, float('nan'))
                    tag = f"V_QPC={vqpc:+.2f} V"
                for xc in XC_SCALES:
                    out = batch_dir / f"case_{idx_case:02d}"
                    out.mkdir(parents=True, exist_ok=True)
                    try:
                        x_val, dN = run_single_data(x_nm, y_nm, V_vals, xc, tag, out)
                        if CORRELATION_ANALYSIS:
                            correlation_pairs.append((x_val, dN))
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
                    if idx_to_vs:
                        triplet = list(idx_to_vs.values())[col_idx]
                        tag = f"VTG={triplet[0]:+.2f} V, V_QPC={triplet[1]:+.2f} V, V_BG={triplet[2]:+.2f} V"
                    else:
                        idx_to_vqpc = _parse_header_vqpc_only(Path(COMBINED_FILE))
                        vqpc = idx_to_vqpc.get(col_idx, float('nan'))
                        tag = f"V_QPC={vqpc:+.2f} V"
                    for xc in XC_SCALES:
                        out = batch_dir / f"case_{idx_case:02d}"
                        out.mkdir(parents=True, exist_ok=True)
                        futures.append((idx_case, tag, xc, ex.submit(run_single_data, x_nm, y_nm, V_vals, xc, tag, out)))
                        idx_case += 1
                for i, tag, xc, fut in futures:
                    try:
                        x_val, dN = fut.result()
                        if CORRELATION_ANALYSIS:
                            correlation_pairs.append((float(x_val), float(dN)))
                        print(f"Finished case {i}: {tag}, XC={xc:.3f}")
                    except Exception as e:
                        print(f"Case {i} failed: {e}")
    else:
        raise ValueError("POTENTIAL_MODE must be 'files' or 'combined'")

    # After completing cases: correlation plot for combined mode using V_QPC on x-axis
    if CORRELATION_ANALYSIS and len(correlation_pairs) >= 1:
        import numpy as _np
        import matplotlib.pyplot as _plt
        xs = _np.array([p[0] for p in correlation_pairs], dtype=float)
        ys = _np.array([p[1] for p in correlation_pairs], dtype=float)
        # Filter NaNs (from files-mode)
        mask = _np.isfinite(xs) & _np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size >= 1:
            if xs.size >= 2:
                a, b = _np.polyfit(xs, ys, 1)
                slope_e_per_v = a  # electrons per volt
                x_line = _np.linspace(xs.min(), xs.max(), 100)
                y_line = a * x_line + b
            else:
                slope_e_per_v = _np.nan
                x_line, y_line = xs, ys
            fig, ax = _plt.subplots(figsize=(5.6, 4.2))
            ax.scatter(xs, ys, color="tab:blue", label="cases")
            if xs.size >= 2:
                ax.plot(x_line, y_line, color="tab:red", label="linear fit")
            ax.set_xlabel("V_QPC [V]")
            ax.set_ylabel("ΔN [electrons]")
            ax.set_title(f"ΔN vs V_QPC | slope={slope_e_per_v:.3e} electrons/V", fontsize=9)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(batch_dir / "correlation_Vqpc_deltaN.png", dpi=240)
            _plt.close(fig)

    print(f"All MAIN2_DOT simulations complete. Results stored in {batch_dir}")


if __name__ == "__main__":
    main()



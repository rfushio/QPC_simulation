import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np

from solvers.solverUCSB_dot import SimulationConfig, ThomasFermiSolver
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# Toggle output directory for Windows
WINDOWS: bool = False

# Grid resolution (square)
GRID_N: int = 32

# Physical / material parameters (inherited by solverUCSB)
BASINHOPPING_NITER: int = 3
BASINHOPPING_STEP_SIZE: float = 0.5
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000

# Magnetic field and magnetic length based domain (total size = 40 l_B)
B_FIELD_T: float = 3  # Tesla

def magnetic_length_m(B_T: float) -> float:
    # l_B = sqrt(h / (2π e B))
    h = 6.62607015e-34
    e = 1.602e-19
    return float(np.sqrt(h / (2.0 * np.pi * e * B_T)))

L_B_M: float = magnetic_length_m(B_FIELD_T)
L_B_M = 7.11e-9
HALF_SPAN_M: float = 20.0 * L_B_M  # half of 40 l_B

# Domain bounds in nm derived from l_B
X_MIN_NM: float = -HALF_SPAN_M * 1e9
X_MAX_NM: float = +HALF_SPAN_M * 1e9
Y_MIN_NM: float = -HALF_SPAN_M * 1e9
Y_MAX_NM: float = +HALF_SPAN_M * 1e9

# Dot gate radius in nm
DOT_RADIUS_NM: float = 70

# Expected bulk filling factor ν (e.g., 0, 1, ...)
EXPECTED_BULK_NU: float = 1.0

# Gate voltages [V]
# Simulate multiple sets (V_in, V_out, V_B) – 3-tuple required per case
V_IN_OUT_PAIRS: list[tuple[float, float, float]] = [
    (-0.3, 0.08, 0.08),
    (-0.275, 0.08, 0.08),
    (-0.25, 0.08, 0.08),
    (-0.225, 0.08, 0.08),
    (-0.2, 0.08, 0.08),
    (-0.175, 0.08, 0.08),
    (-0.15, 0.08, 0.08)
]

# Back-gate voltage [V] (kept or compatibility; not used when running cases)
V_B: float = 0.107
# Thickness of the top BN and bottom BN[nm]
D_T: float = 25.0
D_B: float = 30.0

# Optional scaling/offset (kept for compatibility; typically keep as-is)
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0

# Optional scaling factor for XC potential
# Support multiple values; all combinations with V_IN_OUT_PAIRS will be run
XC_SCALES: list[float] = [1.5]

# Progressive refinement (Matryoshka) toggle
MATRYOSHKA: bool = False

# Parallel execution across V_NS_EW_PAIRS
PARALLEL: bool = True
MAX_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)

# Correlation analysis toggle
CORRELATION_ANALYSIS: bool = False


# -----------------------------------------------------------------------------
# Helper: build V_t(r) for a circular dot
# -----------------------------------------------------------------------------

def build_vt_grid(
    N: int,
    x_min_nm: float,
    x_max_nm: float,
    y_min_nm: float,
    y_max_nm: float,
    dot_radius_nm: float,
    V_in: float,
    V_out: float,
) -> np.ndarray:
    """Create top-gate voltage map V_t on an N×N grid (in volts).

    A central circular dot of radius `dot_radius_nm` is created.
    - Inside the dot -> V_in
    - Outside the dot -> V_out
    """

    x_nm = np.linspace(x_min_nm, x_max_nm, N)
    y_nm = np.linspace(y_min_nm, y_max_nm, N)
    X_nm, Y_nm = np.meshgrid(x_nm, y_nm, indexing="ij")

    # Distance from the center
    R_nm = np.sqrt(X_nm**2 + Y_nm**2)

    # Dot mask
    dot_mask = (R_nm <= dot_radius_nm)

    Vt = np.full((N, N), V_out, dtype=float)
    Vt[dot_mask] = V_in

    return Vt


# -----------------------------------------------------------------------------
# Single-run helper
# -----------------------------------------------------------------------------

def run_single(V_in: float, V_out: float, V_B_case: float, xc_scale: float, out_dir: Path) -> tuple[float, float]:
    # Build Vt grid first
    vt_grid = build_vt_grid(
        GRID_N,
        X_MIN_NM,
        X_MAX_NM,
        Y_MIN_NM,
        Y_MAX_NM,
        DOT_RADIUS_NM,
        V_in,
        V_out,
    )

    # Plot V_t grid and save (axes in l_B units)
    extent_lB = (
        X_MIN_NM / (L_B_M * 1e9),
        X_MAX_NM / (L_B_M * 1e9),
        Y_MIN_NM / (L_B_M * 1e9),
        Y_MAX_NM / (L_B_M * 1e9),
    )
    plt.figure(figsize=(5.5, 5))
    im = plt.imshow(vt_grid.T, origin="lower", extent=extent_lB, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, label="V_t [V]")
    plt.title(f"V_t grid (V_in={V_in:+.3f} V, V_out={V_out:+.3f} V, V_B={float(V_B_case):+.3f} V)")
    plt.xlabel("x [l_B]")
    plt.ylabel("y [l_B]")
    plt.tight_layout()
    plt.savefig(out_dir / "vt_grid.png", dpi=220)
    plt.close()

    cfg = SimulationConfig(
        Nx=GRID_N,
        Ny=GRID_N,
        B=B_FIELD_T,
        V_B=float(V_B_case),
        dt=D_T*1e-9,
        db=D_B*1e-9,
        Vt_grid=vt_grid,
        v_in=float(V_in),
        v_out=float(V_out),
        x_min_nm=X_MIN_NM,
        x_max_nm=X_MAX_NM,
        y_min_nm=Y_MIN_NM,
        y_max_nm=Y_MAX_NM,
        niter=BASINHOPPING_NITER,
        step_size=BASINHOPPING_STEP_SIZE,
        lbfgs_maxiter=LBFGS_MAXITER,
        lbfgs_maxfun=LBFGS_MAXFUN,
        potential_scale=POTENTIAL_SCALE,
        potential_offset=POTENTIAL_OFFSET,
        exc_file="data/0-data/Exc_data_new2.csv",
        solver_type="solverUCSB",
        exc_scale=float(xc_scale),
        use_matryoshka=MATRYOSHKA,
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

    title_extra = (
        f"V_in={V_in:+.3f} V, V_out={V_out:+.3f} V, V_B={float(V_B_case):+.3f} V, "
        f"XC={float(xc_scale):.3f}, ΔN={delta_N:+.2f}"
    )
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

    return float(V_in), float(delta_N)


def main() -> None:
    # Prepare output directories
    base_dir = Path("analysis_dot_folder_windows" if WINDOWS else "analysis_dot_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / (timestamp + "_UCSB_DOT")
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Collect (x, y) pairs for correlation plot
    correlation_pairs: list[tuple[float, float]] = []

    if not PARALLEL:
        idx = 0
        for entry in V_IN_OUT_PAIRS:
            if len(entry) != 3:
                raise ValueError("Each entry in V_IN_OUT_PAIRS must be a 3-tuple: (V_in, V_out, V_B)")
            V_in, V_out, V_Bi = entry
            for xc_scale in XC_SCALES:
                pot_dir = batch_dir / f"case_{idx:02d}"
                pot_dir.mkdir(parents=True, exist_ok=True)
                try:
                    x_val, dN = run_single(V_in, V_out, V_Bi, xc_scale, pot_dir)
                    if CORRELATION_ANALYSIS:
                        correlation_pairs.append((x_val, dN))
                    print(f"Finished UCSB case {idx}: V_in={V_in:+.2f}, V_out={V_out:+.2f}, V_B={V_Bi:+.3f}, XC={xc_scale:.3f}")
                except Exception as e:
                    print(f"UCSB case {idx} failed: {e}")
                idx += 1
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            idx = 0
            for entry in V_IN_OUT_PAIRS:
                if len(entry) != 3:
                    raise ValueError("Each entry in V_IN_OUT_PAIRS must be a 3-tuple: (V_in, V_out, V_B)")
                V_in, V_out, V_Bi = entry
                for xc_scale in XC_SCALES:
                    pot_dir = batch_dir / f"case_{idx:02d}"
                    pot_dir.mkdir(parents=True, exist_ok=True)
                    futures.append((idx, V_in, V_out, V_Bi, xc_scale, executor.submit(run_single, V_in, V_out, V_Bi, xc_scale, pot_dir)))
                    idx += 1
            for i, V_in, V_out, V_Bi, xc_scale, fut in futures:
                try:
                    x_val, dN = fut.result()
                    if CORRELATION_ANALYSIS:
                        correlation_pairs.append((float(x_val), float(dN)))
                    print(f"Finished UCSB case {i}: V_in={V_in:+.2f}, V_out={V_out:+.2f}, V_B={V_Bi:+.3f}, XC={xc_scale:.3f}")
                except Exception as e:
                    print(f"UCSB case {i} failed: {e}")

    # Correlation analysis: x = V_in, y = ΔN
    if CORRELATION_ANALYSIS and len(correlation_pairs) >= 1:
        xs = np.array([p[0] for p in correlation_pairs], dtype=float)
        ys = np.array([p[1] for p in correlation_pairs], dtype=float)

        if xs.size >= 2:
            a, b = np.polyfit(xs, ys, 1)
            slope_e_per_v = a  # electrons per volt
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = a * x_line + b
        else:
            slope_e_per_v = float("nan")
            x_line, y_line = xs, ys

        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.scatter(xs, ys, color="tab:blue", label="cases")
        if xs.size >= 2:
            ax.plot(x_line, y_line, color="tab:red", label="linear fit")
        ax.set_xlabel("V_in [V]")
        ax.set_ylabel("ΔN [electrons]")
        ax.set_title(f"ΔN vs V_in | slope={slope_e_per_v:.3e} electrons/V", fontsize=9)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(batch_dir / "correlation_Vin_deltaN.png", dpi=240)
        plt.close(fig)

    print(f"All UCSB simulations complete. Results stored in {batch_dir}")


if __name__ == "__main__":
    main()



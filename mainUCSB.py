import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import numpy as np

from solvers.solverUCSB import SimulationConfig, ThomasFermiSolver
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


# -----------------------------------------------------------------------------
# USER CONFIGURATION
# -----------------------------------------------------------------------------

# Grid resolution (square)
GRID_N: int = 32

# Physical / material parameters (inherited by solverUCSB)
BASINHOPPING_NITER: int = 5
BASINHOPPING_STEP_SIZE: float = 0.1
LBFGS_MAXITER: int = 1000
LBFGS_MAXFUN: int = 2_000_000

# Magnetic field and magnetic length based domain (total size = 40 l_B)
B_FIELD_T: float = 3.0  # Tesla

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

# X-shaped bar width in nm
BAR_WIDTH_NM: float = 70
# Gate voltages [V]
# Simulate multiple sets (V_NS, V_EW, V_B) – 3-tuple required per case
# Generate 100 cases: V_NS ∈ [-0.15, -0.05] (10 pts), V_EW ∈ [0.04, 0.17] (10 pts), with V_B = -V_NS
_vns_values = np.linspace(-0.020, -0.0, 10)
_vew_values = np.linspace(0.040, 0.070, 10)
#vb_values = np.linspace(0.060, 0.090, 20)
#vew=0.090
#vns=0.0
vb=0.085
V_NS_EW_PAIRS: list[tuple[float, float, float]] = [
    (float(vns), float(vew), float(vb))
    for vns in _vns_values
    for vew in _vew_values
    #for vb in _vb_values
]
# Back-gate voltage [V] (kept for compatibility; not used when running cases)
V_B: float = 0.107
# Thickness of the top BN and bottom BN[nm]
D_T: float = 30.0 
D_B: float = 30.0

# Optional scaling/offset (kept for compatibility; typically keep as-is)
POTENTIAL_SCALE: float = 1.0
POTENTIAL_OFFSET: float = 0.0

# Optional scaling factor for XC potential
XC_SCALE: float = 1.8

# Progressive refinement (Matryoshka) toggle
MATRYOSHKA: bool = False

# Parallel execution across V_NS_EW_PAIRS
PARALLEL: bool = True
MAX_WORKERS: int = max(1, (os.cpu_count() or 2) - 1)


# -----------------------------------------------------------------------------
# Helper: build V_t(r) for an X-shaped mask leaving four triangles
# -----------------------------------------------------------------------------

def build_vt_grid(
    N: int,
    x_min_nm: float,
    x_max_nm: float,
    y_min_nm: float,
    y_max_nm: float,
    bar_width_nm: float,
    V_NS: float,
    V_EW: float,
) -> np.ndarray:
    """Create top-gate voltage map V_t on an N×N grid (in volts).

    The domain is a square [x_min_nm, x_max_nm] × [y_min_nm, y_max_nm] in nm.
    Two diagonal bars of width `bar_width_nm` (centered at the origin, angles ±45°)
    are removed. The remaining four triangular regions receive voltages:
    - Top & bottom (north/south) triangles → V_NS
    - Left & right (east/west) triangles → V_EW
    """

    x_nm = np.linspace(x_min_nm, x_max_nm, N)
    y_nm = np.linspace(y_min_nm, y_max_nm, N)
    X_nm, Y_nm = np.meshgrid(x_nm, y_nm, indexing="ij")

    # Distance to diagonals y = x and y = -x
    sqrt2 = np.sqrt(2.0)
    d1 = np.abs(Y_nm - X_nm) / sqrt2  # distance to y = x
    d2 = np.abs(Y_nm + X_nm) / sqrt2  # distance to y = -x

    # Bars are within half-width to either diagonal
    half_w = bar_width_nm / 2.0
    bar_mask = (d1 <= half_w) | (d2 <= half_w)

    # Remaining triangles are outside the X-shaped bars
    remaining = ~bar_mask

    # Partition remaining region into NS vs EW using |y| vs |x|
    ns_region = remaining & (np.abs(Y_nm) > np.abs(X_nm))
    ew_region = remaining & (np.abs(X_nm) > np.abs(Y_nm))

    Vt = np.zeros((N, N), dtype=float)
    Vt[ns_region] = V_NS
    Vt[ew_region] = V_EW

    return Vt


# -----------------------------------------------------------------------------
# Single-run helper
# -----------------------------------------------------------------------------

def run_single(V_NS: float, V_EW: float, V_B_case: float, out_dir: Path) -> None:
    # Build Vt grid first
    vt_grid = build_vt_grid(
        GRID_N,
        X_MIN_NM,
        X_MAX_NM,
        Y_MIN_NM,
        Y_MAX_NM,
        BAR_WIDTH_NM,
        V_NS,
        V_EW,
        
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
    plt.title(f"V_t grid (V_NS={V_NS:+.3f} V, V_EW={V_EW:+.3f} V, V_B={float(V_B_case):+.3f} V)")
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
        v_ns=float(V_NS),
        v_ew=float(V_EW),
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
        exc_file="data/0-data/Exc_data_new.csv",
        solver_type="solverUCSB",
        exc_scale=XC_SCALE,
        use_matryoshka=MATRYOSHKA,
    )

    solver = ThomasFermiSolver(cfg)

    # Save parameters pre-run
    with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
        for k, v in asdict(cfg).items():
            f.write(f"{k} = {v}\n")

    t0 = time.time()
    solver.optimise()
    exec_sec = time.time() - t0

    title_extra = f"V_NS={V_NS:+.3f} V, V_EW={V_EW:+.3f} V, V_B={float(V_B_case):+.3f} V"
    solver.plot_results(save_dir=str(out_dir), title_extra=title_extra, show=False)
    solver.save_results(out_dir)

    with (out_dir / "simulation_parameters.txt").open("a", encoding="utf-8") as f:
        f.write("\n# Execution time\n")
        f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
        f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")


def main() -> None:
    # Prepare output directories
    base_dir = Path("analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / (timestamp + "_UCSB")
    batch_dir.mkdir(parents=True, exist_ok=True)

    if not PARALLEL:
        for i, entry in enumerate(V_NS_EW_PAIRS):
            if len(entry) != 3:
                raise ValueError("Each entry in V_NS_EW_PAIRS must be a 3-tuple: (V_NS, V_EW, V_B)")
            V_NS, V_EW, V_Bi = entry
            pot_dir = batch_dir / f"case_{i:02d}"
            pot_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_single(V_NS, V_EW, V_Bi, pot_dir)
                print(f"Finished UCSB case {i}: V_NS={V_NS:+.2f}, V_EW={V_EW:+.2f}, V_B={V_Bi:+.3f}")
            except Exception as e:
                print(f"UCSB case {i} failed: {e}")
    else:
        futures = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for i, entry in enumerate(V_NS_EW_PAIRS):
                if len(entry) != 3:
                    raise ValueError("Each entry in V_NS_EW_PAIRS must be a 3-tuple: (V_NS, V_EW, V_B)")
                V_NS, V_EW, V_Bi = entry
                pot_dir = batch_dir / f"case_{i:02d}"
                pot_dir.mkdir(parents=True, exist_ok=True)
                futures.append((i, V_NS, V_EW, V_Bi, executor.submit(run_single, V_NS, V_EW, V_Bi, pot_dir)))
            for i, V_NS, V_EW, V_Bi, fut in futures:
                try:
                    fut.result()
                    print(f"Finished UCSB case {i}: V_NS={V_NS:+.2f}, V_EW={V_EW:+.2f}, V_B={V_Bi:+.3f}")
                except Exception as e:
                    print(f"UCSB case {i} failed: {e}")

    print(f"All UCSB simulations complete. Results stored in {batch_dir}")


if __name__ == "__main__":
    main()



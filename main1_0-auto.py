import numpy as np
from datetime import datetime
from pathlib import Path
from solver1 import SimulationConfig, ThomasFermiSolver
import re
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_sequential_sim(james_file: str = "data/1-data/James.txt", max_potentials: int | None = None):
    """Run simulations for each potential column found in *james_file*.

    Parameters
    ----------
    james_file : str
        Path to the combined potential file containing x, y, z, V1, V2, ... columns.
    max_potentials : int | None
        If given, limit the number of potential columns simulated (useful for quick tests).
    """

    # Load combined data (skip metadata lines beginning with '%')
    data = np.loadtxt(james_file, comments="%")

    # Extract spatial coordinates (nm) and potential columns
    x_nm = data[:, 0]
    y_nm = data[:, 1]
    # z_nm = data[:, 2]  # ignored for 2D simulation
    V_columns = data[:, 3:]

    n_pot = V_columns.shape[1]
    if max_potentials is not None:
        n_pot = min(n_pot, max_potentials)

    # ------------------------------------------------------------------
    # Helper to parse header mapping idx -> (VQPC, VSG) in volts
    # ------------------------------------------------------------------
    def parse_header(james_path: Path):
        """Return dict idx -> (V_QPC, V_SG) from metadata header."""
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

    def make_dir_name(vqpc: float, vsg: float) -> str:
        """Return directory-friendly formatted name like VQPC_m0.40_VSG_m1.20."""
        return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")

    james_path = Path(james_file)
    idx_to_vs = parse_header(james_path)

    # ------------------------------------------------------------------
    # Prepare date-based output directory (analysis_folder/<YYYYMMDD>/)
    # ------------------------------------------------------------------
    base_dir = Path("analysis_folder")
    today_dir = base_dir / datetime.now().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    # Create a single run folder inside today's directory to hold all potentials
    run_ts = datetime.now().strftime("%H%M%S")
    run_dir = today_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper function to run one simulation (must be top-level so it can
    # be pickled by multiprocessing).
    # ------------------------------------------------------------------

    def _run_single_simulation(idx: int,
                               x_nm: np.ndarray,
                               y_nm: np.ndarray,
                               V_vals: np.ndarray,
                               out_dir_str: str,
                               title_extra: str) -> int:
        """Run one Thomas-Fermi simulation and save results.

        Returns the idx so the parent can report progress.
        """

        cfg = SimulationConfig(
            potential_data=(x_nm, y_nm, V_vals),
            exc_file="data/0-data/Exc_data_digitized.csv",
            niter=1,
            lbfgs_maxiter=1000,
            lbfgs_maxfun=200000,
            Nx=128,
            Ny=128,
            potential_scale=1.0,
            potential_offset=0.033,  
        )

        solver = ThomasFermiSolver(cfg)

        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write simulation parameters BEFORE running simulation
        from dataclasses import asdict
        with (out_dir / "simulation_parameters.txt").open("w", encoding="utf-8") as f:
            for k, v in asdict(cfg).items():
                f.write(f"{k} = {v}\n")

        # Run simulation and time it
        import time
        t0 = time.time()
        solver.optimise()
        exec_sec = time.time() - t0

        # Plot results and manually save outputs (avoid overwriting parameters)
        solver.plot_results(save_dir=str(out_dir), title_extra=title_extra)

        np.savez_compressed(
            out_dir / "results.npz",
            **{
                "nu_opt": solver.nu_opt,
                "nu_smoothed": solver.nu_smoothed,
                "Phi": solver.Phi,
                "x": solver.x,
                "y": solver.y,
                **({"energy_history": np.array(solver.energy_history)} if hasattr(solver, "energy_history") else {}),
            }
        )

        with (out_dir / "optimisation.txt").open("w", encoding="utf-8") as f:
            f.write(str(solver.optimisation_result))

        with (out_dir / "execution_time.txt").open("w", encoding="utf-8") as f:
            f.write(f"execution_time_seconds = {exec_sec:.6f}\n")
            f.write(f"execution_time_minutes = {exec_sec/60:.6f}\n")

        return idx

    # ---------------- Parallel execution ----------------
    tasks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str, str]] = []

    for idx in range(n_pot):
        V_vals = V_columns[:, idx]

        # Determine directory name / title for this potential
        if idx in idx_to_vs:
            vqpc, vsg = idx_to_vs[idx]
            dir_name = make_dir_name(vqpc, vsg)
            title_extra = f"V_QPC={vqpc:+.2f} V, V_SG={vsg:+.2f} V"
        else:
            dir_name = f"pot{idx}"
            title_extra = f"Potential idx {idx}"

        out_dir = run_dir / dir_name
        tasks.append((idx, x_nm, y_nm, V_vals, str(out_dir), title_extra))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_run_single_simulation, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                finished_idx = fut.result()
                print(f"Finished potential {finished_idx+1}/{n_pot}.")
            except Exception as e:
                print(f"Simulation failed: {e}")

    print(f"All {n_pot} simulations complete. Results stored in {run_dir}.")


if __name__ == "__main__":
    run_sequential_sim() 
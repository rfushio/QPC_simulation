import numpy as np
from datetime import datetime
from pathlib import Path
from solver import SimulationConfig, ThomasFermiSolver


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

    # Prepare date-based output directory (analysis_folder/<YYYYMMDD>/)
    base_dir = Path("analysis_folder")
    today_dir = base_dir / datetime.now().strftime("%Y%m%d")
    today_dir.mkdir(parents=True, exist_ok=True)

    # Create a single run folder inside today's directory to hold all potentials
    run_ts = datetime.now().strftime("%H%M%S")
    run_dir = today_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(n_pot):
        V_vals = V_columns[:, idx]
        
        # Build simulation configuration using in-memory potential data
        cfg = SimulationConfig(
            potential_data=(x_nm, y_nm, V_vals),  # in nm units
            exc_file="data/0-data/Exc_data_digitized.csv",
            niter=5,  # adjust as needed
            lbfgs_maxiter=1000,
            lbfgs_maxfun=100000,
            Nx=128,
            Ny=128,
        )

        solver = ThomasFermiSolver(cfg)
        solver.optimise()

        # Each potential gets its own sub-folder within the single run directory
        out_dir = run_dir / f"pot{idx}"
        solver.plot_results(save_dir=str(out_dir))
        solver.save_results(str(out_dir))

        print(f"Finished potential #{idx+1}/{n_pot}. Results saved in {out_dir}.")


if __name__ == "__main__":
    run_sequential_sim() 
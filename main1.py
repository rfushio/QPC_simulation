from solver import SimulationConfig, ThomasFermiSolver
from datetime import datetime
from pathlib import Path
import numpy as np

idx = 0 #index of the potential to simulate in 1-data/James.txt

def main():
    # Create configuration (adjust filenames / parameters as needed)

    # Extract spatial coordinates (nm) and potential columns
    data = np.loadtxt("data/1-data/James.txt", comments="%")
    x_nm = data[:, 0]
    y_nm = data[:, 1]
    # z_nm = data[:, 2]  # ignored for 2D simulation
    V_columns = data[:, 3:]
    V_vals = V_columns[:, idx]

    cfg = SimulationConfig(
        # External data files now live under data/
        potential_data=(x_nm, y_nm, V_vals),

        niter=1,  # keep small for demo purposes
        lbfgs_maxiter=1000,
        lbfgs_maxfun=100000,
        Nx=128,
        Ny=128,
        n_potentials=1, #number of potentials to simulate in 1-data/James.txt

        exc_file="data/0-data/Exc_data_digitized.csv",


        )

    solver = ThomasFermiSolver(cfg)
    solver.optimise()

    # Prepare date-based folder under analysis_folder
    base_dir = Path("analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    # Move existing timestamped runs for today into date_dir
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{today}_") and p.parent != date_dir:
            p.rename(date_dir / p.name)

    # Create new timestamped output directory under today's folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = date_dir / timestamp
    solver.plot_results(save_dir=str(out_dir))
    solver.save_results(str(out_dir))


if __name__ == "__main__":
    main() 
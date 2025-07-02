from solver import SimulationConfig, ThomasFermiSolver
from datetime import datetime


def main():
    # Create configuration (adjust filenames / parameters as needed)
    cfg = SimulationConfig(
        # External data files now live under data/
        potential_file="data/0-data/VNS=2.1.txt",
        exc_file="data/0-data/Exc_data_digitized.csv",
        niter=1,  # keep small for demo purposes
        lbfgs_maxiter=1000,
        lbfgs_maxfun=100000,
        Nx=128,
        Ny=128,
    )

    solver = ThomasFermiSolver(cfg)
    solver.optimise()

    # Save into analysis_folder/<timestamp>/ to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"analysis_folder/{timestamp}"
    solver.plot_results(save_dir=out_dir)
    solver.save_results(out_dir)


if __name__ == "__main__":
    main() 
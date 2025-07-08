import numpy as np
from datetime import datetime
from pathlib import Path
from solver import SimulationConfig, ThomasFermiSolver
import re


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

    for idx in range(n_pot):
        V_vals = V_columns[:, idx]
        
        # Build simulation configuration using in-memory potential data
        cfg = SimulationConfig(
            potential_data=(x_nm, y_nm, V_vals),  # in nm units
            exc_file="data/0-data/Exc_data_digitized.csv",
            niter=1,  # adjust as needed
            lbfgs_maxiter=1000,
            lbfgs_maxfun=200000,
            Nx=128,
            Ny=128,
            potential_scale=1.0,
            potential_offset=0.033,  
        )

        solver = ThomasFermiSolver(cfg)
        solver.optimise()

        # Determine output directory name and title with gate voltages, if available
        if idx in idx_to_vs:
            vqpc, vsg = idx_to_vs[idx]
            dir_name = make_dir_name(vqpc, vsg)
            title_extra = f"V_QPC={vqpc:+.2f} V, V_SG={vsg:+.2f} V"
        else:
            dir_name = f"pot{idx}"
            title_extra = f"Potential idx {idx}"

        out_dir = run_dir / dir_name
        # Plot and save with descriptive title
        solver.plot_results(save_dir=str(out_dir), title_extra=title_extra)
        solver.save_results(str(out_dir))

        print(f"Finished potential #{idx+1}/{n_pot}. Results saved in {out_dir}.")


if __name__ == "__main__":
    run_sequential_sim() 
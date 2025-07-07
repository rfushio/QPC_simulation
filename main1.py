from solver import SimulationConfig, ThomasFermiSolver
from datetime import datetime
from pathlib import Path
import numpy as np
import re

# ------------------------------------------------------------
# Specify the potentials you want to simulate by physical gate voltages.
# Give one or more (V_QPC, V_SG) tuples in volts.  Example:
#     desired_pairs = [(-0.40, -1.35), (-0.40, -1.20)]
# ------------------------------------------------------------

desired_pairs = [(0.80, -0.90)]  # <-- EDIT THIS LIST


def parse_header(james_path: Path):
    """Return mapping idx → (VQPC, VSG) floats from the header of *James.txt*."""
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

def main():
    # Create configuration (adjust filenames / parameters as needed)

    # Extract spatial coordinates (nm) and potential columns
    data = np.loadtxt("data/1-data/James.txt", comments="%")
    x_nm = data[:, 0]
    y_nm = data[:, 1]
    # z_nm = data[:, 2]  # ignored for 2D simulation
    V_columns = data[:, 3:]

    # Map requested (V_QPC, V_SG) pairs to column indices ------------------
    james_path = Path("data/1-data/James.txt")
    idx_to_vs = parse_header(james_path)

    # tolerance for float comparison (absolutes in volts)
    tol = 1e-6
    idxs: list[int] = []
    for pair in desired_pairs:
        found = [i for i, vs in idx_to_vs.items() if abs(vs[0] - pair[0]) < tol and abs(vs[1] - pair[1]) < tol]
        if not found:
            raise ValueError(f"Requested pair {pair} not found in James.txt header.")
        idxs.append(found[0])

    # ---------------------------------------------------------------------

    # Prepare date-based parent folder only once
    base_dir = Path("analysis_folder")
    today = datetime.now().strftime("%Y%m%d")
    date_dir = base_dir / today
    date_dir.mkdir(parents=True, exist_ok=True)

    # Move existing timestamped runs for today into date_dir (once per script run)
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{today}_") and p.parent != date_dir:
            p.rename(date_dir / p.name)

    # Create new timestamped parent directory for *this* batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = date_dir / timestamp
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Loop over the requested potential indices
    for idx in idxs:
        V_vals = V_columns[:, idx]

        cfg = SimulationConfig(
            potential_data=(x_nm, y_nm, V_vals),
            B=11.0,
            niter=1,
            lbfgs_maxiter=1000,
            lbfgs_maxfun=100000,
            Nx=64,
            Ny=64,
            n_potentials=len(idxs),
            potential_scale=1.0,
            potential_offset=0.033,  

            exc_file="data/0-data/Exc_data_digitized.csv",
        )

        solver = ThomasFermiSolver(cfg)
        solver.optimise()

        # Each result goes into pot{idx}/ under the batch directory
        pot_dir = batch_dir / f"pot{idx}"

        pair = list(idx_to_vs.values())[idx]
        title_extra = f"V_QPC={pair[0]:+.2f} V, V_SG={pair[1]:+.2f} V"
        solver.plot_results(save_dir=str(pot_dir), title_extra=title_extra)
        solver.save_results(str(pot_dir))


if __name__ == "__main__":
    main() 
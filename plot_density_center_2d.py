import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def parse_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return mapping pot_index -> (VQPC, VSG) in volts."""
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    header_text = " ".join(header_lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    mapping: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(header_text):
        idx = int(m.group(1)) - 1
        mapping[idx] = (float(m.group(2)), float(m.group(3)))
    return mapping


def center_density(npz_path: Path) -> float:
    data = np.load(npz_path)
    x = data["x"]
    y = data["y"]
    nu = data["nu_smoothed"]
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    return float(nu[ix, iy])

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def plot_density_2d(run_dir: Path, james_path: Path, save_path: Path | None = None):
    idx_map = parse_header(james_path)

    vqpc_vals: List[float] = []
    vsg_vals: List[float] = []
    dens_vals: List[float] = []

    for idx, (vqpc, vsg) in idx_map.items():
        npz_file = run_dir / f"pot{idx}" / "results.npz"
        if not npz_file.exists():
            continue
        dens = center_density(npz_file)
        vqpc_vals.append(vqpc)
        vsg_vals.append(vsg)
        dens_vals.append(dens)

    if not vqpc_vals:
        print("No data found – check paths.")
        return

    # Create a regular grid covering the measured (V_QPC, V_SG) space
    vqpc_min, vqpc_max = min(vqpc_vals), max(vqpc_vals)
    vsg_min, vsg_max = min(vsg_vals), max(vsg_vals)

    # Define grid (increase resolution as desired)
    grid_x, grid_y = np.meshgrid(
        np.linspace(vqpc_min, vqpc_max, 200),
        np.linspace(vsg_min, vsg_max, 200),
    )

    # Interpolate / average scattered data onto grid
    grid_z = griddata(
        points=(np.array(vqpc_vals), np.array(vsg_vals)),
        values=np.array(dens_vals),
        xi=(grid_x, grid_y),
        method="cubic",
    )

    # Optional: fallback fill for NaNs (nearest)
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z[nan_mask] = griddata(
            (np.array(vqpc_vals), np.array(vsg_vals)),
            np.array(dens_vals),
            (grid_x[nan_mask], grid_y[nan_mask]),
            method="nearest",
        )

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        grid_z,
        extent=(vqpc_min, vqpc_max, vsg_min, vsg_max),
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    plt.xlabel("V_QPC [V]")
    plt.ylabel("V_SG [V]")
    plt.title("Centre density (ν) across gate voltages – interpolated map")
    cbar = plt.colorbar(im)
    cbar.set_label("ν (center density)")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    run = Path("analysis_folder/20250702/run_152740")
    james = Path("data/1-data/James.txt")
    out = run / "center_density_scatter.png"
    plot_density_2d(run, james, save_path=out) 
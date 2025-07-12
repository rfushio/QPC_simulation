import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_xy_linecut(npz_path: Path, save_path: Path | None = None):
    """Plot ν(x) linecut at y ≈ 0 extracted from *npz_path*.

    Parameters
    ----------
    npz_path : Path
        Path to the `results.npz` file produced by the solver.
    save_path : Path | None, optional
        Where to save the PNG. If *None*, saves alongside the NPZ using
        the filename ``linecut_x.png``.
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    x_m = data["x"]
    y_m = data["y"]
    nu = data["nu_smoothed"]

    # Choose the row closest to y = 0 and the column closest to x = 0
    y_idx = int(np.argmin(np.abs(y_m)))
    x_idx = int(np.argmin(np.abs(x_m)))

    x_nm = x_m * 1e9  # convert to nm for readability
    y_nm = y_m * 1e9  # nm

    line_x = nu[x_idx, :]
    line_y = nu[:, y_idx]

    plt.figure(figsize=(8, 6))
    plt.plot(y_nm, line_y, color="tab:red")
    plt.plot(x_nm, line_x, color="tab:blue")

    plt.xlabel("x,y [nm]")
    plt.ylabel("ν (density)")
    plt.title("X-axis line cut of ν (y ≈ 0) and Y-axis line cut of ν (x ≈ 0)")
    plt.legend(["X-axis line cut", "Y-axis line cut"])
    plt.tight_layout()

    if save_path is None:
        save_path = npz_path.with_name("linecut_xy.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path.resolve()}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_linecut_x.py <path/to/results.npz> [output.png]")
        sys.exit(1)
    npz_file = Path(sys.argv[1])
    out_file: Path | None = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_xy_linecut(npz_file, out_file)


if __name__ == "__main__":
    main() 
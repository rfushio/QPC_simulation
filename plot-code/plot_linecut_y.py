import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_y_linecut(npz_path: Path, save_path: Path | None = None):
    """Plot ν(y) linecut at x ≈ 0 extracted from *npz_path*.

    The density array is assumed to be stored under the key ``nu_smoothed`` and
    the coordinate arrays under ``x`` and ``y`` (in metres). The routine finds
    the column with x closest to 0, extracts that ν(y) profile, converts the y
    axis to nanometres, and plots / saves the result.
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    x_m = data["x"]
    y_m = data["y"]
    nu = data["nu_smoothed"]

    # Choose the column closest to x = 0
    x_idx = int(np.argmin(np.abs(x_m)))

    y_nm = y_m * 1e9  # nm
    line = nu[x_idx, :]

    plt.figure(figsize=(8, 6))
    plt.plot(y_nm, line, color="tab:red")
    plt.xlabel("y [nm]")
    plt.ylabel("ν (density)")
    plt.title("Y-axis line cut of ν (x ≈ 0)")
    plt.tight_layout()

    if save_path is None:
        save_path = npz_path.with_name("linecut_y.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path.resolve()}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_linecut_y.py <path/to/results.npz> [output.png]")
        sys.exit(1)
    npz_file = Path(sys.argv[1])
    out_file: Path | None = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_y_linecut(npz_file, out_file)


if __name__ == "__main__":
    main() 
"""plot_james_sampling.py

Visualise the spatial sampling points contained in `data/1-data/James.txt`.

Usage
-----
python plot-code/plot_james_sampling.py [output.png]

If *output.png* is omitted, the script shows the figure interactively; otherwise
it saves the scatter plot to the specified path.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JAMES_PATH = Path("data/1-data/James.txt")


def load_sampling_points(james_path: Path = DEFAULT_JAMES_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_nm, y_nm) arrays of sampling coordinates from *james_path*."""
    if not james_path.exists():
        raise FileNotFoundError(james_path)

    data = np.loadtxt(james_path, comments="%", usecols=(0, 1))
    x_nm, y_nm = data[:, 0], data[:, 1]
    return x_nm, y_nm


def plot_sampling(x_nm: np.ndarray, y_nm: np.ndarray, save_path: Path | None = None) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(x_nm, y_nm, s=6, color="tab:blue", alpha=0.8, edgecolors="none")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    plt.title("James.txt sampling points (1538 total)")
    plt.axis("equal")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved scatter plot to {save_path.resolve()}")
    else:
        plt.show()

    plt.close()


def main() -> None:
    out_file: Path | None = None
    if len(sys.argv) > 1:
        out_file = Path(sys.argv[1])

    x_nm, y_nm = load_sampling_points()
    plot_sampling(x_nm, y_nm, out_file)


if __name__ == "__main__":
    main() 
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_potential_gradient(npz_path: Path, save_path: Path | None = None) -> None:
    """Visualise first spatial derivatives of the external potential Φ(x, y).

    The function loads *npz_path* (created by the Thomas–Fermi solver), computes
    the gradients ∂Φ/∂x and ∂Φ/∂y using central finite differences, and saves a
    PNG with the two derivative maps side-by-side.

    Parameters
    ----------
    npz_path : Path
        Path to the ``results.npz`` file.
    save_path : Path | None, optional
        Output PNG path. If *None*, the image is saved next to the NPZ under
        the name ``phi_gradient.png``.
    """
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)

    # External potential (Φ), shape (Nx, Ny)
    phi = data["Phi"]

    # Grid coordinates (metres)
    x_m: np.ndarray = data["x"]
    y_m: np.ndarray = data["y"]

    # Ensure 1D coordinate arrays
    if x_m.ndim != 1 or y_m.ndim != 1:
        raise ValueError("x and y arrays in results.npz must be 1-D vectors.")

    # Finite-difference spacing (assumed uniform)
    dx = float(np.mean(np.diff(x_m)))
    dy = float(np.mean(np.diff(y_m)))

    # Compute gradients (V / m)
    dphi_dx, dphi_dy = np.gradient(phi, dx, dy, edge_order=2)

    # Gradient magnitude |∇Φ| (V / m)
    grad_mag = np.sqrt(dphi_dx ** 2 + dphi_dy ** 2)

    # Convert to V / nm for readability
    grad_mag_nm = grad_mag * 1e-9

    # Plotting extent (nm)
    extent = (
        x_m.min() * 1e9,
        x_m.max() * 1e9,
        y_m.min() * 1e9,
        y_m.max() * 1e9,
    )

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(
        grad_mag_nm.T,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=float(np.percentile(grad_mag_nm, 99)),  # robust upper bound
        aspect="auto",
    )
    ax.set_title("|∇Φ| [V/nm]")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("|∇Φ| [V/nm]")

    # Determine output path
    if save_path is None:
        save_path = npz_path.with_name("phi_gradient_abs.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved gradient magnitude figure to {save_path.resolve()}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_potential_gradient.py <path/to/results.npz> [output.png]")
        sys.exit(1)

    npz_file = Path(sys.argv[1])
    out_file: Path | None = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    plot_potential_gradient(npz_file, out_file)


if __name__ == "__main__":
    main() 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RUN_DIR = Path("analysis_folder/20250702/run_152740")  # change if needed


def plot_fixed_range(folder: Path):
    npz_path = folder / "results.npz"
    if not npz_path.exists():
        return False
    data = np.load(npz_path)
    nu = data["nu_smoothed"]
    x = data["x"] * 1e9  # nm
    y = data["y"] * 1e9  # nm

    extent = (x.min(), x.max(), y.min(), y.max())

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        nu,
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    plt.title("ν(r) with fixed color scale [0,1]")
    plt.xlabel("x [nm]")
    plt.ylabel("y [nm]")
    cbar = plt.colorbar(im)
    cbar.set_label("ν")
    plt.tight_layout()

    out_png = folder / "nu_fixed.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved {out_png.relative_to(RUN_DIR)}")
    return True


def main():
    if not RUN_DIR.exists():
        print(f"Run directory {RUN_DIR} not found.")
        return
    count = 0
    for sub in RUN_DIR.iterdir():
        if sub.is_dir():
            if plot_fixed_range(sub):
                count += 1
    print(f"Generated fixed-range plots for {count} subdirectories.")


if __name__ == "__main__":
    main() 
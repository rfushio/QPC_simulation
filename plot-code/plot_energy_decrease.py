import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    """Plot the energy decrease per basinhopping iteration.

    Usage
    -----
    python plot_energy_decrease.py <path/to/results.npz> [output.png]

    The results.npz file must contain an ``energy_history`` array that was
    stored by the solver (one value per *accepted* basinhopping iteration
    plus the initial energy before any optimisation steps).
    """
    if len(sys.argv) < 2:
        print("Usage: python plot_energy_decrease.py <path/to/results.npz> [output.png]")
        sys.exit(1)

    npz_path = Path(sys.argv[1]).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} does not exist")

    out_path: Path | None = None
    if len(sys.argv) >= 3:
        out_path = Path(sys.argv[2]).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path)
    if "energy_history" not in data:
        raise KeyError(
            "energy_history array not found in results.npz. "
            "Please re-run the simulation with an updated solver that records it."
        )

    energy = np.asarray(data["energy_history"], dtype=float)
    # Skip the 0th (initial) energy value for plotting
    if energy.size < 2:
        raise ValueError("energy_history must contain at least two entries (initial + one iteration)")

    energy_full = energy  # keep for ΔE computation
    energy = energy_full[1:]          # use iterations ≥1
    iterations = np.arange(1, len(energy_full))  # 1, 2, ...

    # Energy decrease relative to previous accepted step (length = len(energy))
    delta_E = -np.diff(energy_full)  # positive for decrease, len = len(full)-1

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(7, 4))

    color_energy = "tab:blue"
    ax1.set_xlabel("Basinhopping iteration")
    ax1.set_ylabel("Energy [meV]", color=color_energy)
    ax1.plot(iterations, energy, marker="o", color=color_energy, label="Energy")
    ax1.tick_params(axis="y", labelcolor=color_energy)

    # Secondary axis for energy decrease per iteration
    ax2 = ax1.twinx()
    color_delta = "tab:red"
    ax2.set_ylabel("Energy decrease [meV]", color=color_delta)
    # Plot ΔE from iteration ≥2 (skip the first transition 0→1)
    ax2.bar(iterations[1:], delta_E[1:], color=color_delta, alpha=0.4, label="ΔEnergy (from iter≥2)")
    ax2.tick_params(axis="y", labelcolor=color_delta)

    fig.tight_layout()
    fig.suptitle("Energy and Energy Decrease per Iteration", y=1.04)

    # Legend handling
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    fig.legend(lines, labels, loc="upper right")

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main() 
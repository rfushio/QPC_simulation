import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_potential_labels(james_path: Path) -> Dict[int, str]:
    """Parse potential parameter labels from the header of *James.txt*.

    Returns
    -------
    dict
        Mapping from column index (starting at 0) to a label string, e.g.
        {0: "VQPC=-4, VSG=-1.5", 1: "VQPC=-4, VSG=-1.35", ...}
    """
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                # reached data section
                break
            header_lines.append(line.strip())

    header_text = " ".join(header_lines)

    pattern = re.compile(
        r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V",
        re.IGNORECASE,
    )

    labels: Dict[int, str] = {}
    for match in pattern.finditer(header_text):
        idx = int(match.group(1)) - 1  # convert 1-based to 0-based
        vqpc = match.group(2)
        vsg = match.group(3)
        labels[idx] = f"VQPC={vqpc} V, VSG={vsg} V"
    return labels


def load_density_line(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_nm, nu_line) where nu_line is nu_smoothed at y ≈ 0."""
    data = np.load(npz_path)
    x = data["x"]  # metres
    y = data["y"]  # metres
    nu = data["nu_smoothed"]  # shape (Nx, Ny)

    # find y index closest to 0
    y_idx = int(np.argmin(np.abs(y)))

    # nu array shape is (Nx, Ny) where first axis is x
    line = nu[:, y_idx]
    x_nm = x * 1e9  # convert to nm for nicer plotting
    return x_nm, line


def make_dir_name(vqpc: float, vsg: float) -> str:
    return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")


def dir_for_idx(run_dir: Path, idx: int, mapping: Dict[int, Tuple[float, float]]) -> Path:
    if idx in mapping:
        name = make_dir_name(*mapping[idx])
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    # fallback to potXXX
    return run_dir / f"pot{idx}"


def parse_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.lstrip().startswith("%"):
                break
            header_lines.append(ln.strip())
    text = " ".join(header_lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    mp: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(text):
        mp[int(m.group(1)) - 1] = (float(m.group(2)), float(m.group(3)))
    return mp


def plot_linecuts(
    run_dir: Path,
    james_path: Path,
    pot_indices: Sequence[int] | None = None,
    save_path: Path | None = None,
):
    labels = parse_potential_labels(james_path)
    params_map = parse_header(james_path)

    if pot_indices is None:
        # infer all pot subfolders in run_dir
        pot_indices = sorted(
            int(p.name.replace("pot", ""))
            for p in run_dir.iterdir()
            if p.is_dir() and p.name.startswith("pot")
        )

    plt.figure(figsize=(8, 6))

    for idx in pot_indices:
        pot_dir = dir_for_idx(run_dir, idx, params_map)
        npz_path = pot_dir / "results.npz"
        if not npz_path.exists():
            print(f"[WARN] missing {npz_path}, skipping.")
            continue
        x_nm, line = load_density_line(npz_path)
        label = labels.get(idx, f"pot{idx}")
        plt.plot(x_nm, line, label=label)

    plt.xlabel("x [nm]")
    plt.ylabel("ν (density line cut)")
    plt.title("Line-cut of ν along y ≈ 0 (multiple potentials)")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path.resolve()}")
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage: python plot_linecuts.py
    date_dir = Path("analysis_folder/20250702/run_152740")
    james_txt = Path("data/1-data/James.txt")
    out_png = date_dir / "linecuts.png"

    # limit to first 20 potentials for readability; set to None for all
    plot_linecuts(date_dir, james_txt, pot_indices=range(20), save_path=out_png) 
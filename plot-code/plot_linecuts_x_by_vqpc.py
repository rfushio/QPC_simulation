import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Helper functions (copied/adapted from plot_linecuts.py)
# -----------------------------------------------------------------------------

def parse_potential_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return mapping idx → (VQPC, VSG) in volts."""
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())

    header_text = " ".join(header_lines)
    pattern = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V")

    mapping: Dict[int, Tuple[float, float]] = {}
    for m in pattern.finditer(header_text):
        idx = int(m.group(1)) - 1  # header indices start at 1
        vqpc = float(m.group(2))
        vsg = float(m.group(3))
        mapping[idx] = (vqpc, vsg)
    return mapping


def load_density_line(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    x = data["x"] * 1e9  # nm
    y = data["y"]
    nu = data["nu_smoothed"]
    y_idx = int(np.argmin(np.abs(y)))
    line = nu[:, y_idx]
    return x, line


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def make_dir_name(vqpc: float, vsg: float) -> str:
    return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")

def dir_for_idx(run_dir: Path, idx: int, mapping: Dict[int, Tuple[float, float]]) -> Path:
    if idx in mapping:
        cand = run_dir / make_dir_name(*mapping[idx])
        if cand.exists():
            return cand
    return run_dir / f"pot{idx}"

def plot_grouped_by_vqpc(run_dir: Path, james_path: Path, save_dir: Path | None = None):
    idx_to_params = parse_potential_header(james_path)

    # Build mapping VQPC → list of (idx, VSG)
    vqpc_map: Dict[float, List[Tuple[int, float]]] = {}
    for idx, (vqpc, vsg) in idx_to_params.items():
        vqpc_map.setdefault(vqpc, []).append((idx, vsg))

    if save_dir is None:
        save_dir = run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for vqpc, pairs in sorted(vqpc_map.items()):
        # sort by VSG ascending
        pairs.sort(key=lambda t: t[1])
        plt.figure(figsize=(8, 6))
        plotted = False
        for idx, vsg in pairs:
            pot_dir = dir_for_idx(run_dir, idx, idx_to_params)
            npz_path = pot_dir / "results.npz"
            if not npz_path.exists():
                print(f"[SKIP] {npz_path} not found")
                continue
            x_nm, line = load_density_line(npz_path)
            plt.plot(x_nm, line, label=f"V_SG={vsg} V")
            plotted = True

        if not plotted:
            plt.close()
            print(f"No data for V_QPC={vqpc} – skipping empty plot")
            continue

        plt.xlabel("x [nm]")
        plt.ylabel("ν (density)")
        plt.title(f"Line cuts at V_QPC={vqpc} V (varying V_SG)")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        outfile = save_dir / f"linecuts_VQPC_{vqpc:+.2f}V.png"
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved {outfile}")


if __name__ == "__main__":
    run_path = Path("analysis_folder/20250711/20250711_213220")
    james_txt = Path("data/1-data/James2.txt")
    plot_grouped_by_vqpc(run_path, james_txt) 
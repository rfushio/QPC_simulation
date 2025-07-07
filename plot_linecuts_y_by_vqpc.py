import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def parse_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return mapping idx -> (VQPC, VSG) in volts."""
    lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            lines.append(line.strip())
    text = " ".join(lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    mapping: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(text):
        idx = int(m.group(1)) - 1
        vqpc = float(m.group(2))
        vsg = float(m.group(3))
        mapping[idx] = (vqpc, vsg)
    return mapping


def load_y_line(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    x = data["x"]
    y = data["y"] * 1e9  # nm
    nu = data["nu_smoothed"]
    x_idx = int(np.argmin(np.abs(x)))  # closest to x=0
    return y, nu[x_idx, :]

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def plot_y_by_vqpc(run_dir: Path, james_path: Path, save_dir: Path | None = None):
    idx_map = parse_header(james_path)
    vqpc_map: Dict[float, List[Tuple[int, float]]] = {}
    for idx, (vqpc, vsg) in idx_map.items():
        vqpc_map.setdefault(vqpc, []).append((idx, vsg))

    if save_dir is None:
        save_dir = run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for vqpc, pairs in sorted(vqpc_map.items()):
        pairs.sort(key=lambda t: t[1])  # sort by VSG
        plt.figure(figsize=(8, 6))
        for idx, vsg in pairs:
            npz_path = run_dir / f"pot{idx}" / "results.npz"
            if not npz_path.exists():
                continue
            y_nm, line = load_y_line(npz_path)
            plt.plot(y_nm, line, label=f"V_SG={vsg} V")
        plt.xlabel("y [nm]")
        plt.ylabel("ν (density)")
        plt.title(f"Y-line cuts at V_QPC={vqpc} V (varying V_SG)")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        out = save_dir / f"y_linecuts_VQPC_{vqpc:+.2f}V.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    run = Path("analysis_folder/20250702/run_152740")
    james = Path("data/1-data/James.txt")
    plot_y_by_vqpc(run, james) 
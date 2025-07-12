import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Parse header to mapping idx -> (VQPC, VSG)

def parse_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    header_text = " ".join(header_lines)
    pattern = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.IGNORECASE)
    mapping: Dict[int, Tuple[float, float]] = {}
    for m in pattern.finditer(header_text):
        idx = int(m.group(1)) - 1
        vqpc = float(m.group(2))
        vsg = float(m.group(3))
        mapping[idx] = (vqpc, vsg)
    return mapping


def load_line(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    x = data["x"] * 1e9  # nm
    y = data["y"] # y-axis
    nu = data["nu_smoothed"] # density
    y_idx = int(np.argmin(np.abs(y))) # find the index of the minimum y value
    return x, nu[:, y_idx]


def make_dir_name(vqpc: float, vsg: float) -> str:
    return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")


def dir_for_idx(run_dir: Path, idx: int, mapping: Dict[int, Tuple[float, float]]) -> Path:
    if idx in mapping:
        cand = run_dir / make_dir_name(*mapping[idx])
        if cand.exists():
            return cand
    return run_dir / f"pot{idx}"


def plot_by_vsg(run_dir: Path, james_path: Path, save_dir: Path | None = None):
    idx_map = parse_header(james_path)
    vsg_map: Dict[float, List[Tuple[int, float]]] = {}
    for idx, (vqpc, vsg) in idx_map.items():
        vsg_map.setdefault(vsg, []).append((idx, vqpc))

    if save_dir is None:
        save_dir = run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for vsg, pairs in sorted(vsg_map.items()):
        pairs.sort(key=lambda t: t[1])  # sort by VQPC
        plt.figure(figsize=(8, 6))
        plotted = False
        for idx, vqpc in pairs:
            pot_dir = dir_for_idx(run_dir, idx, idx_map)
            npz_file = pot_dir / "results.npz"
            if not npz_file.exists():
                print(f"[SKIP] {npz_file} not found")
                continue
            x_nm, line = load_line(npz_file)
            plt.plot(x_nm, line, label=f"V_QPC={vqpc} V")
            plotted = True

        if not plotted:
            plt.close()
            print(f"No data for V_SG={vsg} – skipping empty plot")
            continue

        plt.xlabel("x [nm]")
        plt.ylabel("ν (density)")
        plt.title(f"X-Line cuts at V_SG={vsg} V (varying V_QPC)")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        out_path = save_dir / f"linecuts_VSG_{vsg:+.2f}V.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    run = Path("analysis_folder/20250711/20250711_213220")
    james = Path("data/1-data/James2.txt")
    plot_by_vsg(run, james) 
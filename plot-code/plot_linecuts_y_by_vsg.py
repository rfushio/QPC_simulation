import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_header(james: Path) -> Dict[int, Tuple[float, float]]:
    lines: List[str] = []
    with james.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            lines.append(line.strip())
    text = " ".join(lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    d: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(text):
        d[int(m.group(1)) - 1] = (float(m.group(2)), float(m.group(3)))
    return d


def load_y_line(npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    dat = np.load(npz)
    x = dat["x"]
    y = dat["y"] * 1e9  # nm
    nu = dat["nu_smoothed"]
    x_idx = int(np.argmin(np.abs(x)))
    return y, nu[x_idx, :]


def make_dir_name(vqpc: float, vsg: float) -> str:
    return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")


def dir_for_idx(run_dir: Path, idx: int, mapping: Dict[int, Tuple[float, float]]) -> Path:
    if idx in mapping:
        cand = run_dir / make_dir_name(*mapping[idx])
        if cand.exists():
            return cand
    return run_dir / f"pot{idx}"


def plot_y_by_vsg(run_dir: Path, james_path: Path, save_dir: Path | None = None):
    idx_map = parse_header(james_path)
    vsg_groups: Dict[float, List[Tuple[int, float]]] = {}
    for idx, (vqpc, vsg) in idx_map.items():
        vsg_groups.setdefault(vsg, []).append((idx, vqpc))

    if save_dir is None:
        save_dir = run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for vsg, lst in sorted(vsg_groups.items()):
        lst.sort(key=lambda t: t[1])  # by VQPC
        plt.figure(figsize=(8, 6))
        for idx, vqpc in lst:
            pot_dir = dir_for_idx(run_dir, idx, idx_map)
            npz = pot_dir / "results.npz"
            if not npz.exists():
                continue
            y_nm, line = load_y_line(npz)
            plt.plot(y_nm, line, label=f"V_QPC={vqpc} V")
        plt.xlabel("y [nm]")
        plt.ylabel("ν (density)")
        plt.title(f"Y-line cuts at V_SG={vsg} V (varying V_QPC)")
        plt.legend(fontsize="small", ncol=2)
        plt.tight_layout()
        out = save_dir / f"y_linecuts_VSG_{vsg:+.2f}V.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    run_dir = Path("analysis_folder/20250702/run_152740")
    james_txt = Path("data/1-data/James.txt")
    plot_y_by_vsg(run_dir, james_txt) 
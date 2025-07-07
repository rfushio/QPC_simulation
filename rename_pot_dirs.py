import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

RUN_DIR = Path("analysis_folder/20250702/run_152740")  # edit as needed
JAMES_TXT = Path("data/1-data/James.txt")


def parse_header(james_path: Path) -> Dict[int, Tuple[float, float]]:
    """Return mapping pot_index -> (VQPC, VSG)."""
    header_lines: List[str] = []
    with james_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.lstrip().startswith("%"):
                break
            header_lines.append(line.strip())
    text = " ".join(header_lines)
    pat = re.compile(r"@\s*(\d+):\s*VQPC=([+-]?\d*\.?\d+)\s*V,\s*VSG=([+-]?\d*\.?\d+)\s*V", re.I)
    mapping: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(text):
        mapping[int(m.group(1)) - 1] = (float(m.group(2)), float(m.group(3)))
    return mapping


def make_dir_name(vqpc: float, vsg: float) -> str:
    return f"VQPC_{vqpc:+.2f}_VSG_{vsg:+.2f}".replace("+", "p").replace("-", "m")


def rename_dirs(run_dir: Path, mapping: Dict[int, Tuple[float, float]]):
    for pot_dir in sorted(run_dir.glob("pot*")):
        if not pot_dir.is_dir():
            continue
        idx_str = pot_dir.name.replace("pot", "")
        try:
            idx = int(idx_str)
        except ValueError:
            print(f"[SKIP] Unrecognised directory {pot_dir.name}")
            continue
        if idx not in mapping:
            print(f"[SKIP] No voltage mapping for index {idx}")
            continue
        new_name = make_dir_name(*mapping[idx])
        new_path = pot_dir.with_name(new_name)
        if new_path.exists():
            print(f"[WARN] Target directory {new_path} already exists – skipping")
            continue
        os.rename(pot_dir, new_path)
        print(f"Renamed {pot_dir.name} → {new_name}")


if __name__ == "__main__":
    idx_map = parse_header(JAMES_TXT)
    rename_dirs(RUN_DIR, idx_map)
    print("Done renaming directories.") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render paired Bloch-sphere images (start & end/target on ONE sphere).

Usage (from repo root):
  .venv\Scripts\activate
  python compiler\render_bloch_pair.py
  python compiler\render_bloch_pair.py --show
  python compiler\render_bloch_pair.py --open-dir

Flags:
  --show     : opens each generated image after saving
  --open-dir : opens the output folder in the OS file explorer
"""

from pathlib import Path
import sys, os
import json, math
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly; we open files via OS after saving
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_JSON = ROOT / "questions.json"
OUTDIR = ROOT / "assets" / "bloch_pair"
OUTDIR.mkdir(parents=True, exist_ok=True)

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0 else (v / n)

def draw_sphere(ax):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.08, linewidth=0.3, edgecolor='k')
    ax.quiver(0,0,0, 1,0,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,1,0, length=1.0, arrow_length_ratio=0.08)
    ax.quiver(0,0,0, 0,0,1, length=1.0, arrow_length_ratio=0.08)
    ax.text(1.05,0,0,'x')
    ax.text(0,1.05,0,'y')
    ax.text(0,0,1.05,'z')
    ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1]); ax.set_zlim([-1.1,1.1])
    ax.set_box_aspect([1,1,1])
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_zticks([-1,0,1])
    ax.view_init(elev=22, azim=35)

def plot_pair(rA: np.ndarray, rB: np.ndarray, title: str, out_path: Path, labelA: str, labelB: str):
    fig = plt.figure(figsize=(5.2,5.2), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    draw_sphere(ax)
    ax.quiver(0,0,0, *unit(rA), length=float(np.linalg.norm(rA)), arrow_length_ratio=0.12, linewidth=2)
    ax.quiver(0,0,0, *unit(rB), length=float(np.linalg.norm(rB)), arrow_length_ratio=0.12, linewidth=2, linestyle='--')
    ax.scatter([rA[0]],[rA[1]],[rA[2]], s=60, marker='o', label=labelA)
    ax.scatter([rB[0]],[rB[1]],[rB[2]], s=60, marker='^', label=labelB)
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def to_vec(c0: str, c1: str) -> np.ndarray:
    a = complex(c0); b = complex(c1)
    x = 2*np.real(a*np.conj(b))
    y = 2*np.imag(a*np.conj(b))
    z = (abs(a)**2 - abs(b)**2).real
    return np.array([float(x), float(y), float(z)], dtype=float)

def open_file(path: Path) -> None:
    try:
        if os.name == "nt":  # Windows
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":  # macOS
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception as e:
        print(f"[WRN] Could not open {path}: {e}")

def open_dir(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')
    except Exception as e:
        print(f"[WRN] Could not open folder {path}: {e}")

def main(argv: List[str]) -> int:
    if not QUESTIONS_JSON.exists():
        print(f"[ERR] {QUESTIONS_JSON} not found. Run compiler\\build_content.py first.")
        return 1

    want_show = "--show" in argv
    want_open_dir = "--open-dir" in argv

    data: List[Dict[str, Any]] = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    out_paths: List[Path] = []
    n = 0
    for q in data:
        qid = q["id"]
        qtype = (q.get("type") or q.get("template","")).lower()
        out_file = OUTDIR / f"{qid}_pair.png"

        if qtype in ("gate_effect","sq_gate_effect"):
            r0 = q.get("bloch_start")
            r1 = q.get("bloch_end")
            if r0 is None and q.get("state_start"):
                r0 = to_vec(*q["state_start"])
            if r0 is not None and r1 is not None:
                plot_pair(np.array(r0), np.array(r1), f"{qid}: start vs end", out_file, "start", "end")
                out_paths.append(out_file); n += 1

        elif qtype in ("build_target","sq_build_target"):
            r0 = q.get("bloch_start") or [0.0,0.0,1.0]  # |0>
            rT = q.get("bloch_target")
            if rT is None and q.get("target_vector") is not None:
                rT = to_vec(q["target_vector"][0], q["target_vector"][1])
            if r0 is not None and rT is not None:
                plot_pair(np.array(r0), np.array(rT), f"{qid}: start vs target", out_file, "start", "target")
                out_paths.append(out_file); n += 1

    print(f"[OK] Wrote {n} paired Bloch images → {OUTDIR}")

    if want_show:
        for p in out_paths:
            open_file(p)

    if want_open_dir:
        open_dir(OUTDIR)

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

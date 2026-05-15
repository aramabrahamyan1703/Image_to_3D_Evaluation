#!/usr/bin/env python3
"""Compact cross-model report from eval3d results.

Reads results.csv from one or more runs and prints a markdown table
comparing them on a curated set of metrics.

Example:
    python report.py runs/eval/triposr runs/eval/sf3d runs/eval/instantmesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# (column, label, higher_is_better, fmt). None higher_is_better -> no winner mark.
METRICS = [
    ("psnr",                  "PSNR (front view) ↑",          True,  "{:.2f}"),
    ("ssim",                  "SSIM ↑",                       True,  "{:.3f}"),
    ("lpips",                 "LPIPS ↓",                      False, "{:.3f}"),
    ("clip_sim_input",        "CLIP-sim vs input ↑",          True,  "{:.3f}"),
    ("silhouette_iou",        "Silhouette IoU ↑",             True,  "{:.3f}"),
    ("clip_mv_pairwise_mean", "Multi-view consistency ↑",     True,  "{:.3f}"),
    ("ref_chamfer_l1",        "Chamfer-L1 (vs GT) ↓",         False, "{:.4f}"),
    ("ref_fscore@0p01",       "F-score @ 1% ↑",               True,  "{:.3f}"),
    ("ref_fscore@0p02",       "F-score @ 2% ↑",               True,  "{:.3f}"),
    ("ref_fscore@0p05",       "F-score @ 5% ↑",               True,  "{:.3f}"),
    ("ref_normal_consistency","Normal consistency (vs GT) ↑", True,  "{:.3f}"),
    ("num_faces",             "Mean #faces",                  None,  "{:,.0f}"),
    ("is_watertight",         "Watertight (%)",               True,  "{:.0%}"),
]


def load_runs(run_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    out = {}
    for d in run_dirs:
        csv = d / "results.csv"
        if not csv.exists():
            print(f"warn: {csv} missing — skipping", file=sys.stderr)
            continue
        out[d.name] = pd.read_csv(csv)
    return out


def stat(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    if df[col].dtype == bool or set(s.unique()).issubset({0.0, 1.0}):
        return float(s.mean())
    return float(s.mean())


def fmt_table(runs: dict[str, pd.DataFrame]) -> str:
    names = list(runs.keys())
    rows: list[list[str]] = []
    for col, label, hib, fmt in METRICS:
        vals = [stat(runs[n], col) for n in names]
        if all(v is None for v in vals):
            continue
        # determine winner
        winner = None
        if hib is not None:
            numeric = [(i, v) for i, v in enumerate(vals) if v is not None]
            if numeric:
                winner = max(numeric, key=lambda x: x[1])[0] if hib else min(numeric, key=lambda x: x[1])[0]
        cells = [label]
        for i, v in enumerate(vals):
            cell = "—" if v is None else fmt.format(v)
            if i == winner:
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(cells)

    n_samples = {n: len(df) for n, df in runs.items()}
    header = ["Metric"] + [f"{n}<br/><sub>n={n_samples[n]}</sub>" for n in names]
    sep = ["---"] + ["---:"] * len(names)
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(sep) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path,
                   help="Run directories (each containing results.csv)")
    p.add_argument("--out", type=Path, default=None,
                   help="Write markdown table to this file as well as stdout")
    args = p.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print("no runs loaded", file=sys.stderr)
        return 1

    table = fmt_table(runs)
    legend = (
        "\n_↑ higher is better, ↓ lower is better. **Bold** = best across models. "
        "F-score thresholds are fractions of the unit bbox diagonal "
        "(both meshes normalized before sampling)._"
    )
    output = "# 3D-Reconstruction Eval — Model Comparison\n\n" + table + "\n" + legend + "\n"
    print(output)
    if args.out:
        args.out.write_text(output)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

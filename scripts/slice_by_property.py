#!/usr/bin/env python3
"""Per-property breakdown of eval3d results.

Joins per-method results.csv files with a hand-labelled property CSV
(thickness / surface / material) and prints one summary table per
(metric, property) pair, plus a single combined markdown report.

Why this exists
---------------
The presentation feedback asked us to provide a *property-based* report so we
can see how each method performs on a specific *type* of object — not only
which is best on average. This script is the implementation.

Example
-------
    python scripts/slice_by_property.py \\
        --runs runs/eval/triposr runs/eval/sf3d runs/eval/instantmesh \\
        --labels data/property_labels.csv \\
        --out runs/eval/by_property.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


# (column, label, higher_is_better, fmt)
METRICS = [
    ("ref_chamfer_l1",         "Chamfer-L1 ↓",        False, "{:.4f}"),
    ("ref_fscore@0p02",        "F-Score @ 2 % ↑",     True,  "{:.3f}"),
    ("ref_normal_consistency", "Normal Consistency ↑", True,  "{:.3f}"),
    ("silhouette_iou",         "Silhouette IoU ↑",    True,  "{:.3f}"),
    ("clip_mv_pairwise_mean",  "Multi-view CLIP ↑",   True,  "{:.3f}"),
    ("lpips",                  "LPIPS ↓",             False, "{:.3f}"),
]

PROPERTIES = ["thickness", "surface", "material"]


def load_runs(run_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for d in run_dirs:
        csv = d / "results.csv"
        if not csv.exists():
            print(f"warn: {csv} missing — skipping", file=sys.stderr)
            continue
        df = pd.read_csv(csv)
        out[d.name] = df
    return out


def join_with_labels(
    runs: dict[str, pd.DataFrame], labels: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name, df in runs.items():
        merged = df.merge(labels, on="sample_id", how="inner")
        if merged.empty:
            print(
                f"warn: no overlap between {name} sample_ids and label set",
                file=sys.stderr,
            )
        out[name] = merged
    return out


def winner_index(values: Iterable[float | None], higher_is_better: bool) -> int | None:
    nums = [(i, v) for i, v in enumerate(values) if v is not None and pd.notna(v)]
    if not nums:
        return None
    return (max if higher_is_better else min)(nums, key=lambda x: x[1])[0]


def metric_table(
    runs: dict[str, pd.DataFrame],
    metric_col: str,
    metric_label: str,
    higher_is_better: bool,
    fmt: str,
    prop: str,
) -> str:
    """One markdown table: rows = property values, cols = methods."""
    methods = list(runs.keys())
    if not methods:
        return ""

    values_per_value = sorted(
        {v for df in runs.values() if prop in df.columns for v in df[prop].dropna()}
    )
    if not values_per_value:
        return ""

    lines: list[str] = []
    head = ["**" + prop.capitalize() + "**", "n"] + methods
    lines.append("| " + " | ".join(head) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(methods) + 1)) + " |")
    for v in values_per_value:
        ns = []
        means: list[float | None] = []
        for m in methods:
            sub = runs[m][runs[m][prop] == v]
            ns.append(len(sub))
            if metric_col in sub.columns:
                col = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
                means.append(float(col.mean()) if not col.empty else None)
            else:
                means.append(None)
        n = ns[0] if len(set(ns)) == 1 else "/".join(str(x) for x in ns)
        w = winner_index(means, higher_is_better)
        cells = [str(v), str(n)]
        for i, val in enumerate(means):
            cell = "—" if val is None else fmt.format(val)
            if i == w:
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return f"### {metric_label} — by {prop}\n\n" + "\n".join(lines) + "\n"


def build_report(
    runs: dict[str, pd.DataFrame],
    labels_path: Path,
    n_labels: int,
) -> str:
    head = (
        "# Per-Property Breakdown\n\n"
        f"Each method's metrics, broken down by **{', '.join(PROPERTIES)}** "
        f"of the input object. Hand-labelled subset of {n_labels} GSO objects "
        f"(see `{labels_path.as_posix()}`). **Bold** = best per row.\n\n"
        "Property values:\n"
        "- *thickness*: `thin` (towel, motherboard, dipper) · `mid` (jenga block, bowl, headset) · `thick` (sneaker, toaster, basket)\n"
        "- *surface*:   `smooth` (porcelain bowl, tape) · `textured` (printed packaging, sequin boot) · `detailed` (sneaker, motherboard, action figure)\n"
        "- *material*:  `matte` (plastic toy, fabric) · `glossy` (porcelain, foil-printed box) · `mixed` (mostly metallic objects)\n"
    )
    sections: list[str] = [head]
    for prop in PROPERTIES:
        sections.append(f"\n## By `{prop}`\n")
        for col, label, hib, fmt in METRICS:
            tbl = metric_table(runs, col, label, hib, fmt, prop)
            if tbl:
                sections.append(tbl)
    return "\n".join(sections)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="Run directories (each containing results.csv)")
    p.add_argument("--labels", type=Path, required=True,
                   help="CSV with columns: sample_id, thickness, surface, material")
    p.add_argument("--out", type=Path, default=None,
                   help="Write markdown to this file (default: stdout)")
    args = p.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print("no runs loaded", file=sys.stderr)
        return 1

    labels = pd.read_csv(args.labels)
    missing = [c for c in (["sample_id"] + PROPERTIES) if c not in labels.columns]
    if missing:
        print(f"labels file missing columns: {missing}", file=sys.stderr)
        return 1

    runs = join_with_labels(runs, labels)
    text = build_report(runs, args.labels, n_labels=len(labels))
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

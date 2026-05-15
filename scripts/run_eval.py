#!/usr/bin/env python3
"""Standalone runner for the eval3d image-to-3D evaluation pipeline.

Example:
    python scripts/run_eval.py \
        --inputs data/inputs \
        --meshes data/meshes/triposr \
        --references data/references \
        --out runs/eval/triposr \
        --views 8 \
        --image-size 512 \
        --device cuda \
        --save-renders
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `eval3d` importable when running directly without `pip install -e .`.
# This script lives in <repo>/scripts/, so the package root is one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval3d.pipeline import EvalConfig, run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_eval.py",
        description="Evaluation pipeline for image-to-3D mesh generation (with optional reference-based metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inputs", type=Path, required=True,
                   help="Directory of input PNGs")
    p.add_argument("--meshes", type=Path, required=True,
                   help="Directory of generated .obj meshes (filenames must match inputs)")
    p.add_argument("--references", type=Path, default=None,
                   help="Directory of reference (ground-truth) .obj meshes "
                        "(stems must match inputs). Enables F-score / Chamfer / "
                        "normal-consistency metrics for samples with a reference.")
    p.add_argument("--ref-samples", type=int, default=100_000,
                   help="Surface samples per mesh used for reference metrics")
    p.add_argument("--ref-f-thresholds", type=float, nargs="+",
                   default=[0.01, 0.02, 0.05],
                   help="Distance thresholds (fraction of unit bbox diagonal if "
                        "normalized) at which to compute precision/recall/F-score")
    p.add_argument("--no-ref-normalize", action="store_true",
                   help="Skip centering+unit-bbox-diagonal normalization before "
                        "reference metrics (use raw coordinates)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for results.csv / results.json / summary.json")
    p.add_argument("--views", type=int, default=8,
                   help="Number of azimuth views for multi-view consistency")
    p.add_argument("--image-size", type=int, default=512,
                   help="Render and input resolution (square)")
    p.add_argument("--elev", type=float, default=15.0,
                   help="Elevation (deg) for multi-view ring")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device")
    p.add_argument("--save-renders", action="store_true",
                   help="Persist rendered views under <out>/renders/<sample_id>/")
    p.add_argument("--no-rembg", action="store_true",
                   help="Skip rembg; assume input PNGs already carry a foreground alpha")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.inputs.is_dir():
        print(f"error: --inputs not found: {args.inputs}", file=sys.stderr)
        return 2
    if not args.meshes.is_dir():
        print(f"error: --meshes not found: {args.meshes}", file=sys.stderr)
        return 2
    if args.references is not None and not args.references.is_dir():
        print(f"error: --references not found: {args.references}", file=sys.stderr)
        return 2

    cfg = EvalConfig(
        inputs_dir=args.inputs,
        meshes_dir=args.meshes,
        references_dir=args.references,
        out_dir=args.out,
        image_size=args.image_size,
        n_views=args.views,
        elev=args.elev,
        device=args.device,
        save_renders=args.save_renders,
        use_rembg=not args.no_rembg,
        ref_n_samples=args.ref_samples,
        ref_f_thresholds=tuple(args.ref_f_thresholds),
        ref_normalize=not args.no_ref_normalize,
    )
    rows = run(cfg)

    n_err = sum(1 for r in rows if "error" in r)
    print(f"done: {len(rows)} samples, {n_err} failed → {args.out}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

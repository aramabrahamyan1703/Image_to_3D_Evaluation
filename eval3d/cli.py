from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import EvalConfig, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eval3d", description="Image-to-3D evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run evaluation on a paired dataset")
    r.add_argument("--inputs", type=Path, required=True, help="Dir of input PNGs")
    r.add_argument("--meshes", type=Path, required=True, help="Dir of generated .obj")
    r.add_argument("--references", type=Path, default=None,
                   help="Dir of reference (ground-truth) .obj for F-score / Chamfer / etc.")
    r.add_argument("--ref-samples", type=int, default=100_000,
                   help="Surface samples per mesh for reference metrics")
    r.add_argument("--ref-f-thresholds", type=float, nargs="+",
                   default=[0.01, 0.02, 0.05],
                   help="Distance thresholds (in unit-bbox-diag units if normalized) for F-score")
    r.add_argument("--no-ref-normalize", action="store_true",
                   help="Skip centering+unit-bbox normalization before reference metrics")
    r.add_argument("--out", type=Path, required=True, help="Output directory")
    r.add_argument("--views", type=int, default=8)
    r.add_argument("--image-size", type=int, default=512)
    r.add_argument("--elev", type=float, default=15.0)
    r.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--save-renders", action="store_true")
    r.add_argument("--no-rembg", action="store_true",
                   help="Skip rembg; assume input PNG already has a foreground alpha")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        cfg = EvalConfig(
            inputs_dir=args.inputs,
            meshes_dir=args.meshes,
            references_dir=args.references,
            out_dir=args.out,
            ref_n_samples=args.ref_samples,
            ref_f_thresholds=tuple(args.ref_f_thresholds),
            ref_normalize=not args.no_ref_normalize,
            image_size=args.image_size,
            n_views=args.views,
            elev=args.elev,
            device=args.device,
            save_renders=args.save_renders,
            use_rembg=not args.no_rembg,
        )
        run(cfg)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .io_utils import (
    Sample,
    alpha_to_mask,
    composite_white,
    load_image_rgba,
    pair_dataset,
)
from .metrics import alignment, geometry, image_fidelity, multiview, reference
from .render import Renderer, load_mesh_p3d
from .segmentation import ensure_rgba_with_mask


@dataclass
class EvalConfig:
    inputs_dir: Path
    meshes_dir: Path
    out_dir: Path
    references_dir: Optional[Path] = None
    image_size: int = 512
    n_views: int = 8
    elev: float = 15.0
    device: str = "auto"
    save_renders: bool = False
    use_rembg: bool = True
    ref_n_samples: int = 100_000
    ref_f_thresholds: tuple = (0.01, 0.02, 0.05)
    ref_normalize: bool = True


def _resolve_device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_models(device: torch.device):
    import lpips
    import open_clip

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    return lpips_model, clip_model, clip_preprocess


def _prepare_input_image(
    sample: Sample, image_size: int, use_rembg: bool
) -> np.ndarray:
    pil = Image.open(sample.png_path)
    if use_rembg:
        pil = ensure_rgba_with_mask(pil)
    else:
        pil = pil.convert("RGBA")
    pil = pil.resize((image_size, image_size), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float32) / 255.0  # RGBA in [0, 1]


def evaluate_sample(
    sample: Sample,
    *,
    renderer: Renderer,
    cfg: EvalConfig,
    lpips_model,
    clip_model,
    clip_preprocess,
    device: torch.device,
) -> Dict:
    input_rgba = _prepare_input_image(sample, cfg.image_size, cfg.use_rembg)
    input_rgb = composite_white(input_rgba)
    input_mask = alpha_to_mask(input_rgba)

    geom = geometry.compute(sample.obj_path)

    mesh = load_mesh_p3d(sample.obj_path, device)
    front_rgba = renderer.render_rgb(mesh, elev=0.0, azim=0.0)
    front_rgb = composite_white(front_rgba)
    front_sil = renderer.render_silhouette(mesh, elev=0.0, azim=0.0)

    mv_rgba = renderer.render_multiview(mesh, n_views=cfg.n_views, elev=cfg.elev)
    mv_rgb = np.stack([composite_white(v) for v in mv_rgba], axis=0)

    fidelity = image_fidelity.compute(
        input_rgb,
        front_rgb,
        lpips_model=lpips_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
    )
    align = alignment.compute(input_mask, front_sil)
    mv = multiview.compute(
        input_rgb,
        mv_rgb,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=device,
    )

    ref_metrics: Dict = {}
    if sample.ref_path is not None:
        ref_metrics = reference.compute(
            sample.obj_path,
            sample.ref_path,
            n_samples=cfg.ref_n_samples,
            f_thresholds=cfg.ref_f_thresholds,
            normalize=cfg.ref_normalize,
        )

    if cfg.save_renders:
        out = cfg.out_dir / "renders" / sample.sample_id
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray((front_rgb * 255).astype(np.uint8)).save(out / "front.png")
        for i, v in enumerate(mv_rgb):
            Image.fromarray((v * 255).astype(np.uint8)).save(out / f"view_{i:02d}.png")

    return {
        "sample_id": sample.sample_id,
        **geom,
        **fidelity,
        **align,
        **mv,
        **ref_metrics,
    }


def run(cfg: EvalConfig) -> List[Dict]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)
    samples = pair_dataset(cfg.inputs_dir, cfg.meshes_dir, cfg.references_dir)
    renderer = Renderer(image_size=cfg.image_size, device=device)
    lpips_model, clip_model, clip_preprocess = _load_models(device)

    rows: List[Dict] = []
    for s in tqdm(samples, desc="eval3d"):
        try:
            row = evaluate_sample(
                s,
                renderer=renderer,
                cfg=cfg,
                lpips_model=lpips_model,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            row = {"sample_id": s.sample_id, "error": str(e)}
        rows.append(row)

    from . import aggregate

    aggregate.write_csv(rows, cfg.out_dir / "results.csv")
    aggregate.write_json(rows, cfg.out_dir / "results.json")
    aggregate.write_summary(rows, cfg.out_dir / "summary.json")
    return rows

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from PIL import Image


def _embed_batch(images_rgb: np.ndarray, clip_model, clip_preprocess, device) -> torch.Tensor:
    pils = [Image.fromarray((im * 255).astype(np.uint8)) for im in images_rgb]
    batch = torch.stack([clip_preprocess(p) for p in pils]).to(device)
    with torch.no_grad():
        feats = clip_model.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feats


def compute(
    input_rgb: np.ndarray,
    multiview_rgbs: np.ndarray,  # (N, H, W, 3) — rendered views, RGB only
    *,
    clip_model,
    clip_preprocess,
    device: torch.device,
) -> Dict[str, float]:
    view_feats = _embed_batch(multiview_rgbs, clip_model, clip_preprocess, device)
    in_feat = _embed_batch(input_rgb[None], clip_model, clip_preprocess, device)

    sims_to_input = (view_feats @ in_feat.T).squeeze(-1).detach().cpu().numpy()
    pair = (view_feats @ view_feats.T).detach().cpu().numpy()
    n = pair.shape[0]
    iu = np.triu_indices(n, k=1)
    pair_sims = pair[iu] if len(iu[0]) else np.array([1.0])

    return {
        "clip_mv_input_mean": float(sims_to_input.mean()),
        "clip_mv_input_std": float(sims_to_input.std()),
        "clip_mv_input_min": float(sims_to_input.min()),
        "clip_mv_pairwise_mean": float(pair_sims.mean()),
        "clip_mv_pairwise_min": float(pair_sims.min()),
    }

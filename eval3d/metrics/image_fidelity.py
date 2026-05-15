from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_tensor_chw(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """RGB float[0,1] HWC → tensor in [-1, 1] NCHW (LPIPS convention)."""
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    t = t * 2.0 - 1.0
    return t.to(device)


def _clip_embed(image_rgb: np.ndarray, clip_model, clip_preprocess, device) -> torch.Tensor:
    from PIL import Image

    pil = Image.fromarray((image_rgb * 255).astype(np.uint8))
    x = clip_preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return feat


def compute(
    input_rgb: np.ndarray,
    render_rgb: np.ndarray,
    *,
    lpips_model,
    clip_model,
    clip_preprocess,
    device: torch.device,
) -> Dict[str, float]:
    """Both inputs are HxWx3 float32 in [0, 1], same shape."""
    assert input_rgb.shape == render_rgb.shape, (input_rgb.shape, render_rgb.shape)

    psnr = float(peak_signal_noise_ratio(input_rgb, render_rgb, data_range=1.0))
    ssim = float(
        structural_similarity(input_rgb, render_rgb, channel_axis=-1, data_range=1.0)
    )

    with torch.no_grad():
        a = _to_tensor_chw(input_rgb, device)
        b = _to_tensor_chw(render_rgb, device)
        lp = float(lpips_model(a, b).item())

    e1 = _clip_embed(input_rgb, clip_model, clip_preprocess, device)
    e2 = _clip_embed(render_rgb, clip_model, clip_preprocess, device)
    clip_sim = float((e1 @ e2.T).item())

    return {"psnr": psnr, "ssim": ssim, "lpips": lp, "clip_sim_input": clip_sim}

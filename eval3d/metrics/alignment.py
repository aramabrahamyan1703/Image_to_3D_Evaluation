from __future__ import annotations

from typing import Dict

import numpy as np


def compute(input_mask: np.ndarray, render_silhouette: np.ndarray) -> Dict[str, float]:
    """Both inputs HxW float in [0, 1]. Threshold at 0.5 for IoU."""
    a = input_mask > 0.5
    b = render_silhouette > 0.5
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = float(inter / union) if union > 0 else 0.0

    a_area = float(a.mean())
    b_area = float(b.mean())
    area_ratio = b_area / a_area if a_area > 0 else float("nan")

    return {
        "silhouette_iou": iou,
        "input_mask_area": a_area,
        "render_mask_area": b_area,
        "mask_area_ratio": area_ratio,
    }

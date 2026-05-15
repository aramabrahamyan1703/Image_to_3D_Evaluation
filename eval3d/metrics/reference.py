from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import trimesh
from scipy.spatial import cKDTree


DEFAULT_F_THRESHOLDS: tuple[float, ...] = (0.01, 0.02, 0.05)
DEFAULT_N_SAMPLES: int = 100_000


def _normalize_unit_bbox_diag(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    bounds = m.bounds
    center = (bounds[0] + bounds[1]) * 0.5
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    if diag < 1e-12:
        return m
    m.apply_translation(-center)
    m.apply_scale(1.0 / diag)
    return m


def _sample_surface(
    mesh: trimesh.Trimesh, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    points, face_idx = trimesh.sample.sample_surface(
        mesh, n, seed=int(rng.integers(0, 2**31 - 1))
    )
    normals = mesh.face_normals[face_idx]
    return np.asarray(points), np.asarray(normals)


def compute(
    pred_path: str | Path,
    ref_path: str | Path,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    f_thresholds: Sequence[float] = DEFAULT_F_THRESHOLDS,
    normalize: bool = True,
    seed: int = 0,
) -> Dict[str, float]:
    pred = trimesh.load(str(pred_path), process=False, force="mesh")
    ref = trimesh.load(str(ref_path), process=False, force="mesh")
    if not isinstance(pred, trimesh.Trimesh) or not isinstance(ref, trimesh.Trimesh):
        raise ValueError("pred and ref must load as Trimesh")

    if normalize:
        pred = _normalize_unit_bbox_diag(pred)
        ref = _normalize_unit_bbox_diag(ref)

    rng = np.random.default_rng(seed)
    p_pts, p_nrm = _sample_surface(pred, n_samples, rng)
    r_pts, r_nrm = _sample_surface(ref, n_samples, rng)

    tree_r = cKDTree(r_pts)
    tree_p = cKDTree(p_pts)
    d_pr, idx_pr = tree_r.query(p_pts, k=1)  # pred -> ref
    d_rp, idx_rp = tree_p.query(r_pts, k=1)  # ref  -> pred

    chamfer_l1 = float(d_pr.mean() + d_rp.mean())
    chamfer_l2 = float((d_pr ** 2).mean() + (d_rp ** 2).mean())
    hausdorff = float(max(d_pr.max(), d_rp.max()))

    out: Dict[str, float] = {
        "ref_chamfer_l1": chamfer_l1,
        "ref_chamfer_l2": chamfer_l2,
        "ref_hausdorff": hausdorff,
    }

    for tau in f_thresholds:
        precision = float((d_pr < tau).mean())
        recall = float((d_rp < tau).mean())
        denom = precision + recall
        f = 2.0 * precision * recall / denom if denom > 0 else 0.0
        key = f"{tau:g}".replace(".", "p")
        out[f"ref_precision@{key}"] = precision
        out[f"ref_recall@{key}"] = recall
        out[f"ref_fscore@{key}"] = f

    cos_pr = np.einsum("ij,ij->i", p_nrm, r_nrm[idx_pr])
    cos_rp = np.einsum("ij,ij->i", r_nrm, p_nrm[idx_rp])
    out["ref_normal_consistency"] = float(
        0.5 * (np.abs(cos_pr).mean() + np.abs(cos_rp).mean())
    )

    return out

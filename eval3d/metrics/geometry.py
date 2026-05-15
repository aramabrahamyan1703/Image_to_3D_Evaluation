from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import trimesh


def _safe_genus(mesh: trimesh.Trimesh) -> float:
    if not mesh.is_watertight:
        return float("nan")
    return float((2 - mesh.euler_number) / 2)


def _triangle_aspect_mean(mesh: trimesh.Trimesh) -> float:
    v = mesh.vertices[mesh.faces]  # (F, 3, 3)
    e0 = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    e1 = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
    e2 = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
    edges = np.stack([e0, e1, e2], axis=1)
    emax = edges.max(axis=1)
    emin = edges.min(axis=1).clip(min=1e-12)
    return float(np.mean(emax / emin))


def _normal_consistency(mesh: trimesh.Trimesh) -> float:
    fa = mesh.face_adjacency
    if len(fa) == 0:
        return float("nan")
    n = mesh.face_normals
    dots = np.einsum("ij,ij->i", n[fa[:, 0]], n[fa[:, 1]])
    return float((dots > 0).mean())


def _duplicate_vertex_frac(mesh: trimesh.Trimesh) -> float:
    n = len(mesh.vertices)
    if n == 0:
        return 0.0
    unique = trimesh.grouping.unique_rows(mesh.vertices)[0]
    return float(1.0 - len(unique) / n)


def compute(mesh_path: str | Path) -> Dict[str, float]:
    mesh: trimesh.Trimesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{mesh_path} did not load as a Trimesh")

    extents = mesh.bounding_box.extents
    bbox_diag = float(np.linalg.norm(extents))
    aspect = float(extents.max() / max(extents.min(), 1e-12))
    area_faces = mesh.area_faces
    deg_frac = float((area_faces < 1e-12).mean()) if len(area_faces) else 0.0

    return {
        "num_vertices": int(mesh.vertices.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "genus": _safe_genus(mesh),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_watertight else float("nan"),
        "bbox_diagonal": bbox_diag,
        "bbox_aspect_ratio": aspect,
        "degenerate_faces_frac": deg_frac,
        "duplicate_vertex_frac": _duplicate_vertex_frac(mesh),
        "triangle_aspect_mean": _triangle_aspect_mean(mesh),
        "normal_consistency": _normal_consistency(mesh),
    }

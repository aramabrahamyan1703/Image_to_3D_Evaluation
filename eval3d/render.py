from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import trimesh
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    BlendParams,
    DirectionalLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


def _normalize_mesh(verts: torch.Tensor) -> torch.Tensor:
    """Center mesh at origin and scale so its bbox diagonal == 1."""
    bb_min = verts.min(dim=0).values
    bb_max = verts.max(dim=0).values
    center = (bb_min + bb_max) * 0.5
    verts = verts - center
    diag = (bb_max - bb_min).norm().clamp(min=1e-8)
    return verts / diag


def _load_verts_faces(mesh_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if mesh_path.suffix.lower() == ".obj":
        verts, faces, _ = load_obj(str(mesh_path), load_textures=False)
        return verts, faces.verts_idx
    tm = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError(f"{mesh_path} did not load as a single Trimesh")
    verts = torch.from_numpy(np.asarray(tm.vertices, dtype=np.float32))
    faces_idx = torch.from_numpy(np.asarray(tm.faces, dtype=np.int64))
    return verts, faces_idx


def load_mesh_p3d(mesh_path: str | Path, device: torch.device) -> Meshes:
    mesh_path = Path(mesh_path)
    verts, faces_idx = _load_verts_faces(mesh_path)
    verts = _normalize_mesh(verts).to(device)
    faces_idx = faces_idx.to(device)
    # Plain white vertex texture (no UVs).
    white = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=white)
    return Meshes(verts=[verts], faces=[faces_idx], textures=textures)


class Renderer:
    def __init__(
        self,
        image_size: int = 512,
        device: torch.device | None = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        camera_dist: float = 2.2,
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.image_size = image_size
        self.camera_dist = camera_dist
        self.bg_color = bg_color

        raster_rgb = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        raster_sil = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * 1e-4,
            faces_per_pixel=50,
        )
        self._raster_rgb = raster_rgb
        self._raster_sil = raster_sil
        self._lights = DirectionalLights(
            device=self.device,
            direction=((0.0, 0.5, 1.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),),
            specular_color=((0.0, 0.0, 0.0),),
        )

    def _camera(self, elev: float, azim: float) -> FoVPerspectiveCameras:
        R, T = look_at_view_transform(dist=self.camera_dist, elev=elev, azim=azim)
        return FoVPerspectiveCameras(R=R, T=T, device=self.device, fov=40.0)

    def render_rgb(self, mesh: Meshes, elev: float, azim: float) -> np.ndarray:
        cam = self._camera(elev, azim)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam, raster_settings=self._raster_rgb),
            shader=SoftPhongShader(
                device=self.device,
                cameras=cam,
                lights=self._lights,
                blend_params=BlendParams(background_color=self.bg_color),
            ),
        )
        with torch.no_grad():
            img = renderer(mesh)[0]  # (H, W, 4)
        return img.detach().cpu().numpy().clip(0.0, 1.0)

    def render_silhouette(self, mesh: Meshes, elev: float, azim: float) -> np.ndarray:
        cam = self._camera(elev, azim)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam, raster_settings=self._raster_sil),
            shader=SoftSilhouetteShader(),
        )
        with torch.no_grad():
            img = renderer(mesh)[0, ..., 3]  # alpha channel
        return img.detach().cpu().numpy().clip(0.0, 1.0)

    def render_multiview(
        self, mesh: Meshes, n_views: int = 8, elev: float = 15.0
    ) -> np.ndarray:
        azims = np.linspace(0.0, 360.0, n_views + 1)[:-1]
        return np.stack([self.render_rgb(mesh, elev, float(a)) for a in azims], axis=0)

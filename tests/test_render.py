from pathlib import Path

import pytest
import trimesh

pytest.importorskip("pytorch3d")
import torch

from eval3d.render import Renderer, load_mesh_p3d


def test_render_box(tmp_path: Path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    p = tmp_path / "box.obj"
    box.export(p)

    device = torch.device("cpu")
    r = Renderer(image_size=128, device=device)
    mesh = load_mesh_p3d(p, device)
    rgba = r.render_rgb(mesh, elev=0.0, azim=0.0)
    assert rgba.shape == (128, 128, 4)
    assert (rgba[..., 3] > 0).any(), "alpha should be non-zero somewhere"

    sil = r.render_silhouette(mesh, elev=0.0, azim=0.0)
    assert sil.shape == (128, 128)
    assert (sil > 0.5).sum() > 100

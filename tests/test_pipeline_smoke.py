from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh
from PIL import Image

pytest.importorskip("pytorch3d")
pytest.importorskip("lpips")
pytest.importorskip("open_clip")

from eval3d.pipeline import EvalConfig, run


def _make_toy_input(path: Path) -> None:
    arr = np.zeros((128, 128, 4), dtype=np.uint8)
    arr[32:96, 32:96, :3] = 200
    arr[32:96, 32:96, 3] = 255
    Image.fromarray(arr, mode="RGBA").save(path)


def test_smoke(tmp_path: Path):
    inputs = tmp_path / "inputs"
    meshes = tmp_path / "meshes"
    out = tmp_path / "out"
    inputs.mkdir()
    meshes.mkdir()

    _make_toy_input(inputs / "obj_a.png")
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(meshes / "obj_a.obj")

    cfg = EvalConfig(
        inputs_dir=inputs,
        meshes_dir=meshes,
        out_dir=out,
        image_size=128,
        n_views=4,
        device="cpu",
        save_renders=False,
        use_rembg=False,
    )
    rows = run(cfg)
    assert len(rows) == 1
    assert "error" not in rows[0], rows[0].get("error")

    df = pd.read_csv(out / "results.csv")
    for col in [
        "num_vertices", "num_faces", "is_watertight", "psnr", "ssim", "lpips",
        "clip_sim_input", "silhouette_iou", "clip_mv_input_mean",
        "clip_mv_pairwise_mean",
    ]:
        assert col in df.columns, col
    assert not df["num_faces"].isna().any()
    assert (out / "summary.json").exists()

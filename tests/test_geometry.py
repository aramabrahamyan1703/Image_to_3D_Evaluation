from pathlib import Path

import trimesh

from eval3d.metrics import geometry


def test_box(tmp_path: Path):
    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    p = tmp_path / "box.obj"
    box.export(p)

    m = geometry.compute(p)
    assert m["is_watertight"] is True
    assert m["is_winding_consistent"] is True
    assert m["genus"] == 0.0
    assert m["num_faces"] == 12
    assert m["num_vertices"] == 8
    assert m["volume"] > 0
    assert m["surface_area"] > 0
    assert 0.0 <= m["normal_consistency"] <= 1.0
    assert m["degenerate_faces_frac"] == 0.0


def test_sphere(tmp_path: Path):
    sph = trimesh.creation.icosphere(subdivisions=2)
    p = tmp_path / "sph.obj"
    sph.export(p)

    m = geometry.compute(p)
    assert m["is_watertight"]
    assert m["genus"] == 0.0
    assert m["normal_consistency"] > 0.95

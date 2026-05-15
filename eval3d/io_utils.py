from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sample:
    sample_id: str
    png_path: Path
    mesh_path: Path
    ref_path: Optional[Path] = None

    # Back-compat alias; some older code referenced `obj_path`.
    @property
    def obj_path(self) -> Path:
        return self.mesh_path


MESH_EXTENSIONS: tuple[str, ...] = (".obj", ".glb", ".gltf", ".ply", ".stl", ".off")


def _index_meshes(directory: Path) -> dict[str, Path]:
    """Map stem -> mesh path. If a stem has multiple files, prefer .obj, then
    the order in MESH_EXTENSIONS."""
    found: dict[str, Path] = {}
    for ext in MESH_EXTENSIONS:
        for p in sorted(directory.glob(f"*{ext}")):
            found.setdefault(p.stem, p)
    return found


def pair_dataset(
    inputs_dir: str | Path,
    meshes_dir: str | Path,
    references_dir: str | Path | None = None,
) -> List[Sample]:
    inputs_dir = Path(inputs_dir)
    meshes_dir = Path(meshes_dir)

    pngs = {p.stem: p for p in sorted(inputs_dir.glob("*.png"))}
    meshes = _index_meshes(meshes_dir)
    common = sorted(set(pngs) & set(meshes))
    if not common:
        raise FileNotFoundError(
            f"No matching (png, mesh) pairs in {inputs_dir} / {meshes_dir} "
            f"(supported mesh extensions: {', '.join(MESH_EXTENSIONS)})"
        )

    refs: dict[str, Path] = {}
    if references_dir is not None:
        refs = _index_meshes(Path(references_dir))

    return [
        Sample(sid, pngs[sid], mesh_path=meshes[sid], ref_path=refs.get(sid))
        for sid in common
    ]


def load_image_rgba(path: str | Path, size: int | None = None) -> np.ndarray:
    """Load PNG → RGBA float32 in [0, 1]. Optionally resize to (size, size)."""
    img = Image.open(path).convert("RGBA")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def composite_white(rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite RGBA over white → RGB float32 in [0, 1]."""
    rgb = rgba[..., :3]
    a = rgba[..., 3:4]
    return rgb * a + (1.0 - a)


def alpha_to_mask(rgba: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (rgba[..., 3] > thresh).astype(np.float32)

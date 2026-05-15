from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session

    return new_session("u2net")


def ensure_rgba_with_mask(image: Image.Image) -> Image.Image:
    """Return an RGBA image whose alpha encodes the foreground mask.

    If the image already has a non-trivial alpha channel, pass through.
    Otherwise run rembg to segment the foreground.
    """
    if image.mode == "RGBA":
        a = np.asarray(image)[..., 3]
        if a.min() < 250:  # already has meaningful alpha
            return image

    from rembg import remove

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    out_bytes = remove(buf.getvalue(), session=_rembg_session())
    return Image.open(io.BytesIO(out_bytes)).convert("RGBA")

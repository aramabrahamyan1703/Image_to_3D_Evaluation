from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def write_csv(rows: List[Dict], path: str | Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(rows: List[Dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(rows, indent=2, default=_json_default))


def write_summary(rows: List[Dict], path: str | Path) -> None:
    df = pd.DataFrame(rows)
    summary: Dict[str, Dict[str, float]] = {}
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = s.dropna().to_numpy(dtype=float)
        if v.size == 0:
            continue
        summary[col] = {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "count": int(v.size),
        }
    Path(path).write_text(json.dumps(summary, indent=2))


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON-serializable: {type(o)}")

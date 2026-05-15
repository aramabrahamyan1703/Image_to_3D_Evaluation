#!/usr/bin/env python3
"""Render eval3d results as a self-contained HTML report.

Usage:
    python report_html.py runs/eval/triposr runs/eval/sf3d runs/eval/instantmesh \
        --out runs/eval/report.html
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


# (column, label, higher_is_better, fmt, description)
GROUPS = [
    ("Image fidelity (front-view render vs. input image)", [
        ("psnr",           "PSNR",        True,  "{:.2f}",
         "Peak signal-to-noise ratio between the rendered front view and the input image. Higher = pixels match more closely."),
        ("ssim",           "SSIM",        True,  "{:.3f}",
         "Structural similarity. Compares luminance, contrast, and structure in local windows. Higher = better."),
        ("lpips",          "LPIPS",       False, "{:.3f}",
         "Learned perceptual distance (AlexNet features). Closer to human judgments of similarity. Lower = better."),
        ("clip_sim_input", "CLIP-sim",    True,  "{:.3f}",
         "Cosine similarity in CLIP image-embedding space. Captures semantic similarity (object identity, category)."),
    ]),
    ("Alignment with input", [
        ("silhouette_iou", "Silhouette IoU", True, "{:.3f}",
         "Intersection-over-Union between the input foreground mask and the rendered front silhouette. Measures how well scale and pose match the input."),
    ]),
    ("Multi-view consistency", [
        ("clip_mv_pairwise_mean", "MV consistency", True, "{:.3f}",
         "Mean CLIP similarity between rendered views around the object. High = the model looks coherent from all sides (no Janus heads, missing backs)."),
    ]),
    ("Geometric accuracy (vs. ground-truth mesh)", [
        ("ref_chamfer_l1",         "Chamfer-L1",      False, "{:.4f}",
         "Mean bidirectional surface-to-surface distance after both meshes are normalized to unit bounding-box diagonal. Lower = closer reconstruction."),
        ("ref_fscore@0p01",        "F-score @ 1%",    True,  "{:.3f}",
         "F1 of precision/recall at threshold τ = 1% of the bbox diagonal. A point is matched if it is within τ of the other surface. Strict tolerance."),
        ("ref_fscore@0p02",        "F-score @ 2%",    True,  "{:.3f}",
         "Same as F@1% but with a moderate tolerance (2% of bbox diagonal)."),
        ("ref_fscore@0p05",        "F-score @ 5%",    True,  "{:.3f}",
         "Lenient tolerance (5% of bbox diagonal). Captures coarse shape agreement."),
        ("ref_normal_consistency", "Normal consistency", True, "{:.3f}",
         "Mean |cos| between matched face normals (both directions). High = surface orientations agree, not just positions."),
    ]),
    ("Mesh quality", [
        ("num_faces",      "Mean #faces",  None, "{:,.0f}",
         "Average triangle count. More faces = denser mesh, higher download/render cost; not necessarily better quality."),
        ("is_watertight",  "Watertight %", True, "{:.0%}",
         "Fraction of meshes that are closed manifolds (no holes, no boundary edges). Required for volume, fluid sim, 3D printing."),
    ]),
]


def stat(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def winner_index(values: list[float | None], higher_is_better: bool | None) -> int | None:
    if higher_is_better is None:
        return None
    nums = [(i, v) for i, v in enumerate(values) if v is not None]
    if not nums:
        return None
    return (max if higher_is_better else min)(nums, key=lambda x: x[1])[0]


def per_model_wins(runs: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """For each model, which metrics did it win?"""
    names = list(runs.keys())
    wins: dict[str, list[str]] = {n: [] for n in names}
    for _, items in GROUPS:
        for col, label, hib, _fmt, _desc in items:
            vals = [stat(runs[n], col) for n in names]
            w = winner_index(vals, hib)
            if w is not None:
                wins[names[w]].append(label)
    return wins


def render(runs: dict[str, pd.DataFrame]) -> str:
    names = list(runs.keys())
    n_per = {n: len(df) for n, df in runs.items()}
    wins = per_model_wins(runs)

    rows_html: list[str] = []
    for group_label, items in GROUPS:
        rows_html.append(
            f'<tr class="grp"><td colspan="{len(names) + 2}">{html.escape(group_label)}</td></tr>'
        )
        for col, label, hib, fmt, desc in items:
            vals = [stat(runs[n], col) for n in names]
            if all(v is None for v in vals):
                continue
            w = winner_index(vals, hib)
            arrow = '<span class="arrow up">↑</span>' if hib is True else \
                    '<span class="arrow down">↓</span>' if hib is False else ''
            cells = []
            for i, v in enumerate(vals):
                cell = "—" if v is None else fmt.format(v)
                cls = "win" if i == w else ""
                cells.append(f'<td class="num {cls}">{cell}</td>')
            rows_html.append(
                f'<tr><th class="metric">{html.escape(label)} {arrow}'
                f'<div class="desc">{html.escape(desc)}</div></th>'
                + "".join(cells) + "</tr>"
            )

    header_cells = "".join(
        f'<th>{html.escape(n)}<div class="n">n = {n_per[n]}</div></th>' for n in names
    )

    summary_cards = ""
    bullets = {
        "triposr": "Best perceptual fidelity. Strong geometry. Balanced choice.",
        "sf3d": "Best input alignment. Lightest meshes. Weakest GT geometry.",
        "instantmesh": "Best GT geometry. Only watertight outputs. Densest meshes.",
    }
    for n in names:
        w = wins.get(n, [])
        tag = bullets.get(n.lower(), "")
        summary_cards += (
            f'<div class="card"><h3>{html.escape(n)}</h3>'
            f'<p class="tag">{html.escape(tag)}</p>'
            f'<div class="winct">{len(w)} wins</div>'
            f'<ul>{"".join(f"<li>{html.escape(x)}</li>" for x in w) or "<li><em>—</em></li>"}</ul>'
            "</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Image-to-3D Eval — Model Comparison</title>
<style>
  :root {{
    --bg: #fafaf8;
    --fg: #1a1a1a;
    --muted: #666;
    --line: #e3e3df;
    --win-bg: #e8f3e8;
    --win-fg: #1f6b1f;
    --accent: #2b5cb0;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font: 14.5px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         color: var(--fg); background: var(--bg);
         margin: 0; padding: 32px 40px; max-width: 1100px; }}
  h1 {{ font-size: 24px; margin: 0 0 6px 0; }}
  h2 {{ font-size: 16px; margin: 28px 0 12px; color: var(--muted);
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .lede {{ color: var(--muted); margin-bottom: 24px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line);
            vertical-align: top; }}
  thead th {{ font-weight: 600; background: #f0f0ec; border-bottom: 2px solid var(--line); }}
  thead th:not(:first-child) {{ text-align: right; }}
  thead .n {{ font-weight: normal; font-size: 11px; color: var(--muted); margin-top: 2px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; font-feature-settings: "tnum"; }}
  td.win {{ background: var(--win-bg); color: var(--win-fg); font-weight: 600; }}
  tr.grp td {{ background: #f6f6f2; font-weight: 600; font-size: 12.5px;
               color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em;
               padding: 10px 12px; border-bottom: 1px solid var(--line); }}
  th.metric {{ font-weight: 500; }}
  .desc {{ font-weight: normal; font-size: 12px; color: var(--muted); margin-top: 2px;
           max-width: 520px; }}
  .arrow {{ font-size: 11px; padding: 1px 5px; border-radius: 3px; margin-left: 4px;
            vertical-align: middle; }}
  .arrow.up {{ background: #e6efe6; color: #1f6b1f; }}
  .arrow.down {{ background: #f0e6e6; color: #8b1f1f; }}
  .cards {{ display: grid; grid-template-columns: repeat({len(names)}, 1fr);
            gap: 14px; margin: 18px 0 8px; }}
  .card {{ border: 1px solid var(--line); background: white; border-radius: 8px;
           padding: 14px 16px; }}
  .card h3 {{ margin: 0 0 4px; font-size: 16px; color: var(--accent); }}
  .card .tag {{ margin: 0 0 10px; color: var(--muted); font-size: 13px; }}
  .card .winct {{ font-size: 12px; color: var(--muted); margin-bottom: 6px;
                   text-transform: uppercase; letter-spacing: 0.05em; }}
  .card ul {{ margin: 0; padding-left: 18px; }}
  .card li {{ font-size: 13px; padding: 1px 0; }}
  footer {{ color: var(--muted); font-size: 12px; margin-top: 24px;
            border-top: 1px solid var(--line); padding-top: 12px; }}
</style>
</head>
<body>
  <h1>Image-to-3D Eval — Model Comparison</h1>
  <p class="lede">100 paired (image, GT-mesh) samples from the Google Scanned Objects subset.
  Best per row in <span style="background:var(--win-bg);color:var(--win-fg);
  padding:1px 6px;border-radius:3px;font-weight:600;">green</span>.</p>

  <h2>At a glance</h2>
  <div class="cards">{summary_cards}</div>

  <h2>Full results</h2>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>

  <footer>
    ↑ = higher is better; ↓ = lower is better.
    F-score thresholds are fractions of the unit bounding-box diagonal (both meshes are
    centered and rescaled to unit diagonal before sampling).
    Reference metrics use 30k–50k surface samples per mesh.
  </footer>
</body>
</html>
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("report.html"))
    args = p.parse_args()

    runs: dict[str, pd.DataFrame] = {}
    for d in args.runs:
        csv = d / "results.csv"
        if not csv.exists():
            print(f"skip {d}: no results.csv")
            continue
        runs[d.name] = pd.read_csv(csv)
    if not runs:
        return 1

    args.out.write_text(render(runs))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

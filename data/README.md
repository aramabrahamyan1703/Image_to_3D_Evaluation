# Dataset

We evaluate on a 100-object subset of the **Google Scanned Objects (GSO)** dataset
(CC BY 4.0, https://research.google/blog/scanned-objects-by-google-research/).
GSO is a collection of 1000+ photogrammetry-captured everyday objects with
clean ground-truth meshes — exactly what we need to score predicted geometry.

## Files

- `eval_subset.txt` — the **100 sample IDs** in our full evaluation. Each ID is
  the GSO category folder name. The corresponding ground-truth mesh is the
  `meshes/model.obj` inside that folder.
- `property_subset.txt` — the **30 sample IDs** that have hand-labelled
  `(thickness, surface, material)` annotations.
- `property_labels.csv` — the labels themselves; see `docs/DATASET.md` for the
  labelling rubric and limitations.

## Reproducing the inputs

The image-to-3D models we evaluate each take a single front-view PNG. To
regenerate inputs from GSO meshes:

1. Download GSO. Each object lives in `<sample_id>/meshes/model.obj`.
2. Render a 512×512 front view with a white background. We used Blender with
   the camera placed at `(0, -1.6, 0.6)` looking at the origin; a directional
   key light from front-up. Any consistent canonical front-view rendering
   works as long as **all three methods see the same input**.
3. Save as `data/inputs/<sample_id>.png`.

The runs in `runs/eval/<method>/` were produced by the original authors'
inference scripts (TripoSR, InstantMesh, Stable Fast 3D); the predicted
meshes (`*.obj`) were then placed in `data/meshes/<method>/<sample_id>.obj`
and evaluated with `scripts/run_eval.py`.

> Predicted meshes and rendered inputs are heavy and are **not** committed.
> Only the metric outputs (`results.csv`, `results.json`, `summary.json`) are
> in version control.

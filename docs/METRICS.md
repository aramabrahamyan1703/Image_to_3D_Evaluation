# Metrics — what every column in `results.csv` means

The pipeline computes five metric families. Each family answers a different
question. Together they tell us whether a predicted mesh *looks* right, *aligns* with
the input, and *matches the underlying geometry* of the ground truth.

We standardise notation:
- $\hat M$ = predicted mesh, $M^*$ = ground-truth mesh.
- $\hat I_v$ = render of $\hat M$ from view $v$; $I_{\text{in}}$ = the input image.
- $\hat S$, $S_{\text{in}}$ = silhouettes (foreground masks) of the front view and the input image.

All meshes are centred and rescaled so that their **bounding-box diagonal equals 1** before
any reference metric is computed (see `eval3d/metrics/reference.py`). All distances below are
therefore in *units of bbox diagonal*; an "F-Score @ 2 %" tolerance means 0.02 of that diagonal.

## 1. Image fidelity (`eval3d/metrics/image_fidelity.py`)

These compare the front-view render $\hat I_0$ to the input image $I_{\text{in}}$. Both have the
same resolution and a white background.

| Key | Name | Direction | What it measures |
|---|---|---|---|
| `psnr` | Peak Signal-to-Noise Ratio | ↑ | Pixel-level error: $\mathrm{PSNR} = 10 \log_{10}(1 / \mathrm{MSE})$. Sensitive to brightness / colour shifts. |
| `ssim` | Structural Similarity | ↑ | Local windowed agreement of luminance, contrast, structure. Robust to small misalignments. |
| `lpips` | LPIPS (AlexNet) | ↓ | Distance in deep-feature space; correlates with human perceptual judgments far better than PSNR/SSIM. |
| `clip_sim_input` | CLIP cosine similarity | ↑ | Cosine of CLIP embeddings of input vs. front render. Captures *semantic* identity (an elephant is still recognised as an elephant even if its texture is slightly off). |

Limit: PSNR/SSIM/LPIPS need both renders to be aligned in the image plane. We render both
through the same renderer with the same camera, so they are.

## 2. Input alignment (`eval3d/metrics/alignment.py`)

| Key | Name | Direction | What it measures |
|---|---|---|---|
| `silhouette_iou` | **Silhouette IoU** | ↑ | $|\hat S \cap S_{\text{in}}| / |\hat S \cup S_{\text{in}}|$. Does the predicted mesh occupy *the same screen region* as the input? Catches scale and pose mismatches that CLIP/PSNR miss because they tolerate uniform shifts. |
| `input_mask_area`, `render_mask_area`, `mask_area_ratio` | mask diagnostics | — | Bookkeeping: whether the mismatch is "render too small" or "input poorly segmented". |

A method can win PSNR while losing Silhouette IoU — the front face matches in colour but the
mesh is too small / off-centre / rotated.

## 3. Multi-view consistency (`eval3d/metrics/multiview.py`)

We render $\hat M$ from $N=8$ azimuths around it and compute CLIP embeddings of all $N$ views.

| Key | Name | Direction | What it measures |
|---|---|---|---|
| `clip_mv_input_mean` | mean CLIP(view, input) | ↑ | Average semantic similarity between every rendered view and the input. High = the object stays recognisable when rotated. |
| `clip_mv_input_min` | worst view | ↑ | The single most-degenerate view's similarity to the input — exposes "Janus" failures (a fine front, a blank back). |
| `clip_mv_pairwise_mean` | mean CLIP(view$_i$, view$_j$) | ↑ | How similar the views are to each other. High = coherent object. |
| `clip_mv_pairwise_min` | worst pairwise | ↑ | Largest jump between two views — front vs. back disagreement. |

This is **reference-free**: no GT mesh required. It's the single best signal of "did the model
hallucinate a coherent 3D object or just a 2.5D painted disk?"

## 4. Geometric accuracy vs. ground truth (`eval3d/metrics/reference.py`)

We sample 100 000 surface points from each of $\hat M$ and $M^*$ (after normalisation), build
KD-trees, and compute nearest-neighbour distances both ways.

Let $d_{p \to r}$ = distance from each predicted sample to its nearest GT sample, and
$d_{r \to p}$ the reverse.

| Key | Name | Direction | Definition |
|---|---|---|---|
| `ref_chamfer_l1` | **Chamfer-L1** | ↓ | $\mathrm{mean}(d_{p \to r}) + \mathrm{mean}(d_{r \to p})$. Penalises both missing parts (extra GT distance) and hallucinated parts (extra pred distance). |
| `ref_chamfer_l2` | Chamfer-L2 | ↓ | Same but squared distances; more sensitive to outliers / spikes. |
| `ref_hausdorff` | Hausdorff | ↓ | $\max(\max d_{p \to r}, \max d_{r \to p})$. Worst-point distance — dominated by isolated artefacts. |
| `ref_precision@τ`, `ref_recall@τ` | Precision/recall at $\tau$ | ↑ | Fraction of pred (resp. GT) samples within $\tau$ of the other surface. |
| `ref_fscore@τ` | **F-Score @ τ** | ↑ | $2 P R / (P + R)$ at thresholds $\tau \in \{0.01, 0.02, 0.05\}$ of the bbox diagonal. The standard 3D-recon metric. |
| `ref_normal_consistency` | **Normal Consistency** (NC) | ↑ | $\tfrac12 (\mathbb E[|\cos\theta_{p \to r}|] + \mathbb E[|\cos\theta_{r \to p}|])$ where $\theta$ is the angle between matched face normals. Measures *surface orientation* — high NC means the surfaces don't just sit close, they face the same way. |

Why both Chamfer and F-Score? Chamfer is a *mean*, so a few outlier vertices blow it up.
F-Score is a *count* under a tolerance, robust to outliers. Either alone is misleading.

## 5. Mesh quality (`eval3d/metrics/geometry.py`)

These are mesh-only — no rendering, no GT.

| Key | Name | What it measures |
|---|---|---|
| `num_vertices`, `num_faces` | size | Bigger ≠ better; expensive to render. |
| `is_watertight` | **W** | Closed manifold (no holes). Required for volume, fluid sim, 3D printing. |
| `is_winding_consistent` | winding | All face normals agree on which side is "outside". |
| `genus`, `euler_number` | topology | Number of handles; only meaningful when `is_watertight`. |
| `surface_area`, `volume`, `bbox_diagonal`, `bbox_aspect_ratio` | size | Sanity-check units. |
| `degenerate_faces_frac`, `duplicate_vertex_frac`, `triangle_aspect_mean` | hygiene | High values flag broken / spiky triangles. |
| `normal_consistency` (mesh-only) | NC (intrinsic) | Local face-adjacency normal agreement — different from `ref_normal_consistency`. |

## How metrics are reported

- **`results.csv`** — one row per sample, all metrics as columns.
- **`results.json`** — same, JSON.
- **`summary.json`** — per-metric aggregate (mean, median, std, min, max, count).
- **`comparison.md` / `report.html`** — cross-method comparison built by `scripts/report*.py`.
- **`by_property.md`** — property-stratified breakdown built by `scripts/slice_by_property.py`.

## Cheat sheet — which metric to trust for which question

| Question | Metric |
|---|---|
| Does the front view *look* like the input? | LPIPS (perceptual), SSIM (structure), PSNR (pixel). |
| Is the object *recognisable* from any angle? | `clip_mv_input_min`. |
| Did the back / sides degenerate? | `clip_mv_pairwise_min`. |
| Is the mesh the right *size and pose*? | `silhouette_iou`. |
| Is the *shape* right? | F-Score @ 2 %, then Chamfer-L1. |
| Are *surface orientations* right? | `ref_normal_consistency`. |
| Can it be 3D-printed / used in a sim? | `is_watertight`. |

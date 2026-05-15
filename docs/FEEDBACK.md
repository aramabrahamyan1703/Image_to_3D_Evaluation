# How this final submission addresses the presentation feedback

## Original feedback

> **For the presentation:**
> 1. The slides contained too much text for a 10-minute presentation.
>    It was hard to follow the slides and the presenter's speech simultaneously.
> 2. The 3D / NeRF related terminology was heavily used in the slides/speech without
>    proper introduction. Since 3D modality was not discussed during the lectures, it
>    would be confusing for the audience.
> 3. Some of the reported metrics are not introduced at all (e.g. NC, W,
>    Silhouette IoU, etc.).
> 4. Although the evaluation & conclusion sections were both individually strong,
>    there is no clear connection how the reported results/metrics implicate the
>    conclusion.
>
> **For the final submission:**
> 5. Make the connection between the reported metrics & conclusion clearer,
>    explaining why a certain approach behaves a certain way according to a certain
>    metric, and why that implicates that it would work better on certain type of
>    objects.
> 6. Provide a property-based report of metrics to see how each approach performs on
>    a specific type of objects (e.g. according to thickness, surface type, material
>    type). If manual labeling takes too long, pick an even smaller subset (and report
>    which data points are used in this smaller evaluation set) from the chosen
>    evaluation set.

## What changed

### (1) Slide density

The new `slides/presentation.tex`:

- Body text on every slide is **≤ 7-word bullets**. Detail moved to the speaker's
  notes (`\note{...}`), which only the speaker sees.
- Removed all paragraph-style prose from non-text-required slides.
- Kept all visuals (architecture diagrams, qualitative renders, scatter plots).

### (2) 3D / NeRF terminology introduced before use

A dedicated **3D primer** slide is inserted between Motivation and Models. It defines:

- **Mesh** = vertices + faces (triangles).
- **Triplane** = three orthogonal 2-D feature grids that compactly encode a 3-D scene.
- **NeRF MLP** = network that maps `(point, view direction) → (density, colour)`.
- **Rendering** = projecting the encoded scene to a 2-D image from a chosen camera.
- **View synthesis** = rendering an unseen view of the same object.

These five terms cover everything the method slides need.

### (3) Every metric is defined before it is used

Two **metrics primer** slides immediately precede the results slides:

- One slide for the **reference-free** group (PSNR / SSIM / LPIPS / CLIP / Sil-IoU /
  multi-view CLIP).
- One slide for the **reference-based** group (Chamfer-L1, F-Score @ τ, Normal
  Consistency, Watertight %).

Each metric gets: (a) the formula in symbols, (b) one plain-language sentence, (c) the
direction (↑ better / ↓ better). In particular:

- **NC** = **Normal Consistency**: $\tfrac12\big(\mathbb E[|\cos\theta_{p\to r}|] + \mathbb E[|\cos\theta_{r\to p}|]\big)$
  — average alignment of matched face normals; high NC means surfaces don't just *touch* the
  GT, they *face the same way*.
- **W %** = **Watertight percentage**: fraction of meshes whose surface is a closed manifold
  with no boundary edges. Required for volume / 3D-printing / fluid-sim downstream.
- **Sil-IoU** = **Silhouette IoU**: $|\hat S \cap S_{\text{in}}| / |\hat S \cup S_{\text{in}}|$
  on the front view — does the predicted mesh occupy the same screen region as the
  input?

Full mathematical write-ups live in [`docs/METRICS.md`](METRICS.md).

### (4) and (5) Metrics → conclusions, explicitly

Each result slide is now structured as **`metric → finding → architectural reason`**.
Examples:

- *F-Score @ 2 % @100*: TripoSR 0.257, **InstantMesh 0.275**, SF3D 0.166 → **InstantMesh wins** because its diffusion stage hallucinates 6 unseen views, recovering occluded back surfaces that single-stage methods can't see.
- *Normal Consistency*: **TripoSR 0.642**, InstantMesh 0.627, SF3D 0.526 → **TripoSR wins** because its NeRF representation produces a single coherent density field; SF3D's vertex-offset trick optimises smoothness *intrinsically* but doesn't guarantee correct *orientation* against GT.
- *Silhouette IoU*: **SF3D 0.271**, InstantMesh 0.119, TripoSR 0.104 → **SF3D wins by 2.5×** because its UV-PBR export pipeline preserves the canonical scale better; the other two routinely produce a mesh that's off-centre or scaled wrong relative to the input frame.
- *Watertight %*: **InstantMesh 38 %**, TripoSR 0 %, SF3D 0 % → **InstantMesh** is the only method whose FlexiCubes iso-surface stage outputs closed manifolds at all; the other two extract via Marching Cubes on a learned density and routinely leave holes.

Each finding is **on the same slide as the number that supports it**, with a coloured
arrow (`Metric ▸ Why ▸ When it matters`).

### (6) Property-based report

Implemented in [`scripts/slice_by_property.py`](../scripts/slice_by_property.py),
output in [`runs/eval/by_property.md`](../runs/eval/by_property.md), and a dedicated
slide in the presentation.

The 30-object hand-labelled subset is documented in [`data/property_labels.csv`](../data/property_labels.csv) and [`docs/DATASET.md`](DATASET.md). Across thickness / surface / material:

- **TripoSR** wins LPIPS on **every** property cell — its perceptual texture transfer is
  consistent regardless of object type. Wins Chamfer-L1 on `mid` and `thin` thickness, on
  `smooth` and `detailed` surface, and on `glossy` and `matte` material — i.e. the typical
  case. **Loses on `mixed` material** (n=3, mostly metallic) — its DINOv1 encoder + 64²
  triplane can't resolve metallic specular cues.
- **InstantMesh** wins F-Score @ 2 % on **`textured` surface** by a large margin
  (0.411 vs. 0.279 / 0.146). Textured = printed-graphics packaging boxes, simple geometry +
  rich visual cues — exactly the regime where its multi-view diffusion shines. Also wins
  NC on **`mixed` material** — closed-manifold output is the reason metal regions don't
  spike.
- **SF3D** wins Silhouette IoU and multi-view CLIP in **every** slice. Material/lighting
  decomposition pays off as *robust input alignment* — its mesh is always the right size
  and pose, even when its absolute geometry is worst.

These are the kind of "*X wins on Y because Z*" claims the original feedback was asking
for, grounded in numbers from the property breakdown.

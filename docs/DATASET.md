# Dataset details

## Why GSO

We needed (a) clean ground-truth meshes, (b) consistent capture conditions, (c) wide
property coverage (thin and thick objects, smooth and detailed surfaces, plastic and
metal and ceramic), and (d) a permissive licence. **Google Scanned Objects** ticks all
four. It's a public CC BY 4.0 dataset of 1000+ photogrammetry-scanned everyday objects —
shoes, kitchenware, electronics, toys.

GSO is reference-grade: each object's `meshes/model.obj` is a watertight, manifold,
metric-scaled scan. Predicted meshes are evaluated *against this directly* (after both
are centred and rescaled to a unit bounding-box diagonal).

## Selection — the 100-object subset

We sampled 100 GSO objects to cover the categories that stress different parts of an
image-to-3D pipeline:

- **Household** (bowls, baskets, kitchen items)
- **Apparel** (shoes, boots, flip-flops)
- **Electronics** (motherboards, laptops, headphones, hard drives)
- **Toys / organic** (action figures, foam dragons, Triceratops, dollhouse parts)
- **Packaging / printed** (DVD cases, supplement boxes, soda 12-packs)

The 100 IDs are listed in [`data/eval_subset.txt`](../data/eval_subset.txt). The same
100 inputs were fed to TripoSR, InstantMesh, and Stable Fast 3D so the comparison is
truly apples-to-apples.

## Property labels — the 30-object subset

Per the project feedback, we need to know *which kind of object* each method handles
best, not only the average score. We hand-labelled a 30-object subset of the 100 along
three axes:

| Axis | Values | Definition |
|---|---|---|
| **thickness** | `thin` / `mid` / `thick` | The *minimum* dimension of the bounding shape, relative to other GSO objects. A motherboard or a flip-flop is `thin`; a Jenga block or a bowl is `mid`; a sneaker, toaster, or basket is `thick`. |
| **surface** | `smooth` / `textured` / `detailed` | `smooth` = mostly featureless geometry (porcelain bowl); `textured` = printed graphics on simple geometry (12-pack box, DVD); `detailed` = lots of surface features in the geometry (sneaker, motherboard, action figure). |
| **material** | `matte` / `glossy` / `mixed` | Dominant photometric character. `mixed` is reserved for objects with significant metallic + plastic regions (e.g. motherboards with metal heat-sinks; metal hammer with painted handle; metal C-clamp). |

The labels are in [`data/property_labels.csv`](../data/property_labels.csv). The 30 IDs
covered are listed in [`data/property_subset.txt`](../data/property_subset.txt).

### Distribution across the 30-object set

```
thickness:  thin  8   mid 12   thick 10
surface:    smooth 11  textured  8   detailed 11
material:   matte 16   glossy 11    mixed     3
```

The `mixed` cell is small (3 samples) by design: there are only a handful of objects
in our 100-set with significant exposed metal. We report the numbers but flag them as
under-powered in the slides.

### Limitations of these labels

- **They are coarse.** A motherboard has thin and thick parts — we labelled it `thin`
  because that's the dominant failure mode for image-to-3D models on it. Single-axis
  labels can't capture every nuance.
- **They are single-rater.** A second annotator might disagree on borderline cases.
- **They are subjective at the bins' boundaries.** A "midsize" Jenga block is *thicker*
  than a flip-flop but *thinner* than a basket; the `mid` bin is doing a lot of work.

We chose 30 (rather than all 100) precisely because the project guidance said: "If
manual labeling takes too long, pick an even smaller subset (and report which data
points are used in this smaller evaluation set) from the chosen evaluation set." 30 is
small enough to be hand-curated carefully, large enough that 8–16 samples land in each
property cell.

## Reproducing on a different dataset

Replace `eval_subset.txt`, regenerate the inputs and predicted meshes, and run
`scripts/run_eval.py` exactly as in the README. The pipeline does not assume GSO
specifically — any directory of paired `(input.png, predicted.obj, gt.obj)` triplets
works.

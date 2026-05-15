# 3D-Reconstruction Eval — Model Comparison

| Metric | triposr<br/><sub>n=100</sub> | sf3d<br/><sub>n=100</sub> | instantmesh<br/><sub>n=100</sub> |
| --- | ---: | ---: | ---: |
| PSNR (front view) ↑ | 11.19 | **11.31** | 11.15 |
| SSIM ↑ | **0.756** | 0.731 | 0.727 |
| LPIPS ↓ | **0.455** | 0.491 | 0.507 |
| CLIP-sim vs input ↑ | 0.706 | **0.708** | 0.680 |
| Silhouette IoU ↑ | 0.104 | **0.271** | 0.119 |
| Multi-view consistency ↑ | 0.903 | **0.917** | 0.897 |
| Chamfer-L1 (vs GT) ↓ | **0.1294** | 0.1852 | 0.1323 |
| F-score @ 1% ↑ | 0.134 | 0.084 | **0.152** |
| F-score @ 2% ↑ | 0.257 | 0.166 | **0.275** |
| F-score @ 5% ↑ | 0.520 | 0.365 | **0.521** |
| Normal consistency (vs GT) ↑ | **0.642** | 0.526 | 0.627 |
| Mean #faces | 130,405 | 26,774 | 446,948 |
| Watertight (%) | 0% | 0% | **38%** |

_↑ higher is better, ↓ lower is better. **Bold** = best across models. F-score thresholds are fractions of the unit bbox diagonal (both meshes normalized before sampling)._

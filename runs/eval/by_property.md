# Per-Property Breakdown

Each method's metrics, broken down by **thickness, surface, material** of the input object. Hand-labelled subset of 30 GSO objects (see `data/property_labels.csv`). **Bold** = best per row.

Property values:
- *thickness*: `thin` (towel, motherboard, dipper) · `mid` (jenga block, bowl, headset) · `thick` (sneaker, toaster, basket)
- *surface*:   `smooth` (porcelain bowl, tape) · `textured` (printed packaging, sequin boot) · `detailed` (sneaker, motherboard, action figure)
- *material*:  `matte` (plastic toy, fabric) · `glossy` (porcelain, foil-printed box) · `mixed` (mostly metallic objects)


## By `thickness`

### Chamfer-L1 ↓ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | **0.1220** | 0.2134 | 0.1402 |
| thick | 10 | 0.1384 | 0.1712 | **0.1378** |
| thin | 8 | **0.1444** | 0.2711 | 0.1498 |

### F-Score @ 2 % ↑ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | **0.353** | 0.153 | 0.315 |
| thick | 10 | 0.245 | 0.181 | **0.326** |
| thin | 8 | **0.231** | 0.099 | 0.219 |

### Normal Consistency ↑ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | **0.639** | 0.512 | 0.594 |
| thick | 10 | 0.635 | 0.560 | **0.656** |
| thin | 8 | 0.587 | 0.460 | **0.619** |

### Silhouette IoU ↑ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | 0.147 | **0.286** | 0.105 |
| thick | 10 | 0.074 | **0.262** | 0.097 |
| thin | 8 | 0.092 | **0.290** | 0.110 |

### Multi-view CLIP ↑ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | 0.900 | **0.919** | 0.896 |
| thick | 10 | 0.904 | **0.917** | 0.891 |
| thin | 8 | 0.906 | **0.920** | 0.896 |

### LPIPS ↓ — by thickness

| **Thickness** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| mid | 12 | **0.395** | 0.445 | 0.448 |
| thick | 10 | **0.454** | 0.491 | 0.515 |
| thin | 8 | **0.395** | 0.416 | 0.454 |


## By `surface`

### Chamfer-L1 ↓ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | **0.1406** | 0.2030 | 0.1519 |
| smooth | 11 | **0.1443** | 0.2194 | 0.1664 |
| textured | 8 | 0.1086 | 0.2245 | **0.0947** |

### F-Score @ 2 % ↑ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | **0.244** | 0.168 | 0.227 |
| smooth | 11 | **0.329** | 0.129 | 0.273 |
| textured | 8 | 0.279 | 0.146 | **0.411** |

### Normal Consistency ↑ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | 0.564 | 0.514 | **0.582** |
| smooth | 11 | **0.657** | 0.537 | 0.608 |
| textured | 8 | 0.661 | 0.483 | **0.695** |

### Silhouette IoU ↑ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | 0.080 | **0.279** | 0.080 |
| smooth | 11 | 0.139 | **0.256** | 0.118 |
| textured | 8 | 0.103 | **0.310** | 0.116 |

### Multi-view CLIP ↑ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | 0.898 | **0.909** | 0.898 |
| smooth | 11 | 0.901 | **0.923** | 0.892 |
| textured | 8 | 0.911 | **0.926** | 0.892 |

### LPIPS ↓ — by surface

| **Surface** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| detailed | 11 | **0.436** | 0.464 | 0.497 |
| smooth | 11 | **0.400** | 0.450 | 0.452 |
| textured | 8 | **0.406** | 0.440 | 0.465 |


## By `material`

### Chamfer-L1 ↓ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | **0.1345** | 0.2117 | 0.1351 |
| matte | 16 | **0.1055** | 0.2167 | 0.1228 |
| mixed | 3 | 0.2785 | **0.2153** | 0.2691 |

### F-Score @ 2 % ↑ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | 0.266 | 0.136 | **0.327** |
| matte | 16 | **0.332** | 0.166 | 0.304 |
| mixed | 3 | 0.097 | 0.093 | **0.110** |

### Normal Consistency ↑ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | 0.625 | 0.545 | **0.636** |
| matte | 16 | **0.647** | 0.499 | 0.619 |
| mixed | 3 | 0.496 | 0.482 | **0.582** |

### Silhouette IoU ↑ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | 0.111 | **0.261** | 0.126 |
| matte | 16 | 0.119 | **0.313** | 0.099 |
| mixed | 3 | 0.032 | **0.166** | 0.047 |

### Multi-view CLIP ↑ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | 0.900 | **0.924** | 0.893 |
| matte | 16 | 0.904 | **0.915** | 0.891 |
| mixed | 3 | 0.908 | **0.914** | 0.912 |

### LPIPS ↓ — by material

| **Material** | n | triposr | sf3d | instantmesh |
| --- | ---: | ---: | ---: | ---: |
| glossy | 11 | **0.428** | 0.474 | 0.481 |
| matte | 16 | **0.378** | 0.413 | 0.439 |
| mixed | 3 | **0.558** | 0.585 | 0.616 |

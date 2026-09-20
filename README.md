# pcp

Small point-cloud utilities built with PyTorch.

The package currently includes:

- a `PointCloud` tensor wrapper
- CSV loading
- farthest-point and random sampling
- k-nearest-neighbor search
- Chamfer distance
- inverse-distance weighted interpolation
- PointNet classification and segmentation models

## Installation

Install the project in editable mode:

```bash
pip install -e .
```

Using `uv`:

```bash
uv pip install -e .
```

Python 3.10 or newer is required.

## Basic usage

```python
import torch

from pcp import PointCloud
from pcp.distance import chamfer
from pcp.grouping import knn
from pcp.sampling import fps, random

cloud = PointCloud(torch.rand(1024, 3))  # (N, C)

sampled = fps(cloud, k=256)
random_sampled = random(cloud, k=256)

center = torch.tensor([0.5, 0.5, 0.5])
neighbors = knn(cloud, center, k=16)

distance = chamfer(sampled, random_sampled)
```

Interpolate scalar or vector values at new coordinates with IDW:

```python
from pcp.interpolation import IDW

known_points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
known_values = torch.tensor([10.0, 20.0, 30.0])
query_points = torch.tensor([[0.25, 0.25]])

interpolated = IDW(
    known_points,
    known_values,
    query_points,
    alpha=2.0,
    n_neighbors=3,
)
```

Set `n_neighbors=None` to use every known point. IDW also supports batched
coordinates `(B, N, C)` and vector values `(B, N, F)`.

Load a point cloud from CSV:

```python
from pcp.io import read_csv

cloud = read_csv("points.csv")
```

## PointNet

PointNet models accept batched tensors in `(B, N, C)` format:

- `B`: batch size
- `N`: number of points
- `C`: number of input channels

```python
import torch

from pcp.nn.PointNet import PointNetCls, PointNetSeg

points = torch.randn(4, 1024, 3)  # (B, N, C)

classifier = PointNetCls(
    in_channels=3,
    num_classes=10,
    use_tnet=True,
)
class_logits = classifier(points)  # (4, 10)

segmenter = PointNetSeg(
    in_channels=3,
    num_classes=4,
    use_tnet=True,
)
point_logits = segmenter(points)  # (4, 4, 1024)
```

An unbatched `PointCloud` with shape `(N, C)` can be prepared for PointNet with:

```python
batched_points = cloud.points.unsqueeze(0)  # (1, N, C)
```

Segmentation logits use `(B, num_classes, N)`, which can be passed directly to PyTorch's `CrossEntropyLoss` with targets shaped `(B, N)`.

## Source layout

```text
src/pcp/
├── core/       # PointCloud
├── distance/   # Chamfer distance
├── grouping/   # k-nearest neighbors
├── interpolation/ # inverse-distance weighting
├── io/         # CSV loading
├── nn/         # PointNet models
└── sampling/   # FPS and random sampling
```

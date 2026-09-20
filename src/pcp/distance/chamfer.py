import torch

from pcp.core.PointCloud import PointCloud


def chamfer(
    points_a: PointCloud | torch.Tensor,
    points_b: PointCloud | torch.Tensor,
) -> torch.Tensor:
    a = points_a.points if isinstance(points_a, PointCloud) else points_a
    b = points_b.points if isinstance(points_b, PointCloud) else points_b

    distances = torch.cdist(a, b)
    return distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean()

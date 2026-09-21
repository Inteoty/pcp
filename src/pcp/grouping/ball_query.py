import torch

from pcp.core.PointCloud import PointCloud


def ball_query(
    radius: float,
    n_samples: int,
    points: PointCloud | torch.Tensor,
    centroids: PointCloud | torch.Tensor,
):
    points = points.points if isinstance(points, PointCloud) else points
    centroids = centroids.points if isinstance(centroids, PointCloud) else centroids

    unbatched = points.ndim == 2 and centroids.ndim == 2
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if centroids.ndim == 2:
        centroids = centroids.unsqueeze(0)

    batch_size, num_points, _ = points.shape
    distances = torch.cdist(centroids, points)

    k = min(n_samples, num_points)
    distances, group_idx = torch.topk(distances, k, dim=-1, largest=False)

    first_idx = group_idx[:, :, :1]
    first_idx = first_idx.repeat(1, 1, k)
    group_idx[distances > radius] = first_idx[distances > radius]

    if k < n_samples:
        padding = group_idx[:, :, :1].repeat(1, 1, n_samples - k)
        group_idx = torch.cat([group_idx, padding], dim=-1)

    batch_idx = torch.arange(batch_size, device=points.device)
    batch_idx = batch_idx.view(batch_size, 1, 1)
    group_points = points[batch_idx, group_idx]

    if unbatched:
        group_idx = group_idx.squeeze(0)
        group_points = group_points.squeeze(0)
    return group_idx, group_points
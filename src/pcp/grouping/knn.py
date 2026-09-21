import torch

from pcp.core.PointCloud import PointCloud

def knn(points: PointCloud | torch.Tensor, centroids: PointCloud | torch.Tensor, k: int) -> PointCloud:
    points = points.points if isinstance(points, PointCloud) else points
    centroids = centroids.points if isinstance(centroids, PointCloud) else centroids

    unbatched = points.ndim == 2 and centroids.ndim <= 2
    if centroids.ndim == 1:
        centroids = centroids.unsqueeze(0)

    if points.ndim == 2:
        points = points.unsqueeze(0)
    if centroids.ndim == 2:
        centroids = centroids.unsqueeze(0)
    distances = torch.cdist(centroids, points)
    _, indices = torch.topk(distances, k, dim=-1, largest=False)

    batch_size = points.shape[0]
    batch_idx = torch.arange(batch_size, device=points.device)
    batch_idx = batch_idx.view(batch_size, 1, 1)
    result = points[batch_idx, indices]

    if unbatched:
        result = result.squeeze(0)

    return PointCloud(result)

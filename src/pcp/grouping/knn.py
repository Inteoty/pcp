import torch
from pcp.core.PointCloud import PointCloud

def knn(points: PointCloud, centroid: PointCloud | torch.Tensor, k: int) -> PointCloud:
    pts = points.points
    C = centroid.points if isinstance(centroid, PointCloud) else centroid
    if C.ndim == 1:
        C = C.unsqueeze(0)

    distances = torch.cdist(C, pts)
    _, indices = torch.topk(distances, k, dim=-1, largest=False)

    result = pts[indices]
    if C.shape[0] == 1:
        result = result.squeeze(0)

    return PointCloud(result)
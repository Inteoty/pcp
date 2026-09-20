import torch
from pcp.core.PointCloud import PointCloud

def random(points: PointCloud, k: int) -> PointCloud:
    pts = points.points
    N = pts.shape[0]

    indices = torch.randperm(N, device=pts.device)[:k]
    return PointCloud(pts[indices])
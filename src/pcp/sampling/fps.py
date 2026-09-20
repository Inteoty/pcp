import torch
from pcp.core.PointCloud import PointCloud

def fps(points: PointCloud, k: int) -> PointCloud:
    pts = points.points
    N = pts.shape[0]

    indices = torch.zeros(k, dtype=torch.long, device=pts.device)
    distances = torch.full((N,), float("inf"), device=pts.device)

    current_idx = 0
    for i in range(k):
        indices[i] = current_idx
        dist = torch.sum((pts - pts[current_idx]) ** 2, dim=-1)
        distances = torch.minimum(distances, dist)
        current_idx = torch.argmax(distances)

    return PointCloud(pts[indices])
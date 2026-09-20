import torch

from pcp.core.PointCloud import PointCloud


def IDW(
    points: PointCloud | torch.Tensor,
    values: torch.Tensor,
    queries: PointCloud | torch.Tensor,
    alpha: float = 2.0,
    n_neighbors: int | None = None,
) -> torch.Tensor:
    source = points.points if isinstance(points, PointCloud) else points
    target = queries.points if isinstance(queries, PointCloud) else queries

    unbatched = source.ndim == 2
    scalar_values = values.ndim == source.ndim - 1

    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)
        values = values.unsqueeze(0)
    if scalar_values:
        values = values.unsqueeze(-1)

    distances = torch.cdist(target, source)
    k = source.shape[1] if n_neighbors is None else n_neighbors
    distances, indices = distances.topk(k, dim=-1, largest=False)

    batch = torch.arange(source.shape[0], device=source.device)[:, None, None]
    neighbor_values = values[batch, indices]

    exact = distances == 0
    weights = distances.clamp_min(torch.finfo(distances.dtype).eps).pow(-alpha)
    weights = torch.where(exact.any(dim=-1, keepdim=True), exact, weights)
    weights = weights / weights.sum(dim=-1, keepdim=True)

    result = (neighbor_values * weights.unsqueeze(-1)).sum(dim=2)
    if scalar_values:
        result = result.squeeze(-1)
    if unbatched:
        result = result.squeeze(0)
    return result

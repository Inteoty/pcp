import torch
import polars as pl

from pcp.core.PointCloud import PointCloud

def read_csv(
    path: str,
    has_header: bool = True,
    delimiter: str = ",",
    device: str | torch.device = "cpu"
) -> PointCloud:
    data = pl.read_csv(path, separator=delimiter, has_header=has_header)
    data = data.to_numpy()
    tensor_data = torch.from_numpy(data).to(device)
    
    return PointCloud(tensor_data)
	
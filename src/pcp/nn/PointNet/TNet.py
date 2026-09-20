import torch
from torch import nn

from pcp.core.PointCloud import PointCloud

class TNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv_layers = nn.Sequential(
            nn.Conv1d(dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.transform = nn.Linear(256, dim * dim)
        nn.init.zeros_(self.transform.weight)
        nn.init.zeros_(self.transform.bias)
    
    def forward(self, x: PointCloud | torch.Tensor):
        x = x.points if isinstance(x, PointCloud) else x
        batch_size = x.shape[0]

        x = self.conv_layers(x)
        x = self.max_pool(x).view(batch_size, -1)
        x = self.fc_layers(x)
        x = self.transform(x)

        identity = torch.eye(
            self.dim,
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        matrix = x.view(batch_size, self.dim, self.dim) + identity
        
        return matrix
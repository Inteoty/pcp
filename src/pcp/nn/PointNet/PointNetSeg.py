import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet.PointNet import PointNetBackbone

class PointNetSeg(nn.Module):
    def __init__(
        self,
		in_channels: int, 
        num_classes: int,
        use_tnet: bool = False
    ):
        super().__init__()
        self.use_tnet = use_tnet

        self.pointnet = PointNetBackbone(
            in_channels = in_channels,
            out_channels = 1024,
            global_feats = False,
            use_tnet = use_tnet
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv1d(1088, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, num_classes, kernel_size=1)
        )

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points

        concat_feats, _, _ = self.pointnet(x)
        logits = self.segmentation_head(concat_feats)
        return logits
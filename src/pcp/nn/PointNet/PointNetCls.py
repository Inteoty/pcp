import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet.PointNet import PointNetBackbone

class PointNetCls(nn.Module):
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
            global_feats = True,
            use_tnet = use_tnet
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)

        )

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points

        global_feats, _ = self.pointnet(x)
        logits = self.classifier(global_feats)
        return logits





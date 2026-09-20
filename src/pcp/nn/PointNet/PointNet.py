import torch
from torch import nn

from pcp.core.PointCloud import PointCloud
from pcp.nn.PointNet.TNet import TNet

class PointNetBackbone(nn.Module):
    def __init__(
		self,
		in_channels: int, 
		out_channels: int,
        global_feats: bool,
        use_tnet: bool = False
	):
        super().__init__()
        self.global_feats = global_feats
        self.use_tnet = use_tnet
        self.shared_mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.max_pool = self.pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size = 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, out_channels, kernel_size = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

        if self.use_tnet:
            self.tnet1 = TNet(dim = in_channels)
            self.tnet2 = TNet(dim = 64) 

    def forward(self, points: PointCloud | torch.Tensor):
        x = points.points if isinstance(points, PointCloud) else points

        batch_size, num_points, _ = x.shape
        # PointCloud uses (B, N, C), while Conv1d expects (B, C, N).
        x = x.transpose(1, 2).contiguous()

        trans_input = None
        if self.use_tnet:
            trans_input = self.tnet1(x)
            x = torch.bmm(x.transpose(2, 1), trans_input).transpose(2, 1)

        x = self.shared_mlp1(x)

        trans_feat = None
        if self.use_tnet:
            trans_feat = self.tnet2(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)

        point_feats = x 

        x = self.shared_mlp2(x)

        global_feats = self.pool(x).view(batch_size, -1) # (B, out_channels)

        # classification
        if self.global_feats:
            # return global_feat, trans_input, trans_feat
            return global_feats, x
        #segmentation
        else:
            global_expanded = global_feats.unsqueeze(-1).repeat(1, 1, num_points)
            concat_feats = torch.cat([point_feats, global_expanded], dim=1)
            return concat_feats, point_feats, global_feats

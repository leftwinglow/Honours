import dgl
from dgl.nn.pytorch import GraphConv
import torch


class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.relu(h)
        h = self.conv2(g, h)
        g.ndata["h"] = h
        return dgl.mean_nodes(g, "h")

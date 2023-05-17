import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import MaxPooling


"""
GIN architecture
from https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
"""

class MLP_GIN(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = F.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class GIN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=5,
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum"):
        super().__init__()

        self.inp_embedding = nn.Embedding(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.inp_transform = nn.Identity()

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        assert aggregator_type in ["sum", "mean", "max"]
        for layer in range(num_layers - 1):  # excluding the input layer
            mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, aggregator_type=aggregator_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_dim = output_dim
        self.graph_level_output = graph_level_output
        # linear functions for graph poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            self.linear_prediction.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim+graph_level_output))
            )
        self.readout = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim+graph_level_output)
        )
        self.drop = nn.Dropout(dropout)
        self.pool = MaxPooling()

    def forward(self, g, state, reward_exp=None):
        assert reward_exp is None

        h = self.inp_embedding(state)
        h = self.inp_transform(h)
        # list of hidden representation at each layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1))

        if self.graph_level_output > 0:
            return score_over_layer[..., :self.output_dim], \
                   self.pool(g, score_over_layer[..., self.output_dim:])
        else:
            return score_over_layer
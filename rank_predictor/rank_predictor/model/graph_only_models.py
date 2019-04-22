import torch
from graph_nets.functions.aggregation import AvgAggregation, MaxAggregation
from graph_nets.block import GNBlock
from graph_nets.data_structures.graph import Graph
from torch import nn, Tensor
from graph_nets.functions.update import NodeAggregationGlobalStateUpdate, IndependentNodeUpdate


class GNAvg(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 1)
        self.core = GNBlock(phi_v=IndependentNodeUpdate(self.dense))
        self.dec = GNBlock(rho_vu=AvgAggregation(), phi_u=NodeAggregationGlobalStateUpdate())

    def forward(self, g: Graph) -> torch.Tensor:
        g: Graph = self.core(g)
        return self.dec(g).attr.val


class GNMax(nn.Module):

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 1)
        self.core = GNBlock(phi_v=IndependentNodeUpdate(self.dense))
        self.dec = GNBlock(rho_vu=MaxAggregation(), phi_u=NodeAggregationGlobalStateUpdate())

    def forward(self, g: Graph) -> torch.Tensor:
        g: Graph = self.core(g)
        return self.dec(g).attr.val

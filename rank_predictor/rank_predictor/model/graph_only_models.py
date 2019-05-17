from copy import deepcopy

import torch
from graph_nets.data_structures.edge import Edge

from graph_nets.functions.aggregation import AvgAggregation, MaxAggregation
from graph_nets.block import GNBlock
from graph_nets.data_structures.graph import Graph
from torch import nn, Tensor
from graph_nets.functions.update import NodeAggregationGlobalStateUpdate, IndependentNodeUpdate
from rank_predictor.model.graph_extractor_full import DecoderGlobalStateUpdate, EncoderEdgeUpdate, \
    EncoderGlobalStateUpdate, CoreGlobalStateUpdate, CoreNodeUpdate, CoreEdgeUpdate
from rank_predictor.model.utils import ListModule


class GNAvg(nn.Module):
    """[baseline+avg] model"""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 1)
        self.core = GNBlock(phi_v=IndependentNodeUpdate(self.dense))
        self.dec = GNBlock(rho_vu=AvgAggregation(), phi_u=NodeAggregationGlobalStateUpdate())

    def forward(self, g: Graph) -> torch.Tensor:
        g: Graph = self.core(g)
        return self.dec(g).attr.val


class GNMax(nn.Module):
    """[baseline+max] model"""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 1)
        self.core = GNBlock(phi_v=IndependentNodeUpdate(self.dense))
        self.dec = GNBlock(rho_vu=MaxAggregation(), phi_u=NodeAggregationGlobalStateUpdate())

    def forward(self, g: Graph) -> torch.Tensor:
        g: Graph = self.core(g)
        return self.dec(g).attr.val


class GNDeep(nn.Module):
    """[n-core(-shared)]"""

    def __init__(self, drop_p: float, num_core_blocks: int, edge_mode: str, shared_weights: bool = False):
        """
        Deep graph network for domain rank estimation.
        :param drop_p: Dropout probability
        :param num_core_blocks: Number of stacked core blocks, >= 0
        :param edge_mode: Whether to keep the graph edges, remove them altogether, or make them bi-directional. In any
                          case, the existence of reflexive edges is ensured.
        :param shared_weights:
        """

        super().__init__()

        self.drop_p = drop_p

        self.edge_fns = {
            'default': GNDeep.default,
            'bi_directional': GNDeep.bi_directional,
            'no_edges': GNDeep.no_edges,
            'all_edges': GNDeep.all_edges
        }

        assert edge_mode in self.edge_fns, "Invalid edge mode; not in [default, bi_directional, no_edges, all_edges]"
        self.edge_mode = edge_mode

        self.enc = GNBlock(
            phi_e=EncoderEdgeUpdate(),
            phi_u=EncoderGlobalStateUpdate(),
            rho_eu=AvgAggregation())

        assert num_core_blocks >= 0

        core_blocks = []
        for i in range(num_core_blocks):
            if shared_weights and i > 0:
                block = core_blocks[0]
            else:
                block = GNBlock(
                    phi_e=CoreEdgeUpdate(self.drop_p),
                    phi_v=CoreNodeUpdate(self.drop_p),
                    phi_u=CoreGlobalStateUpdate(self.drop_p),
                    rho_ev=AvgAggregation(),
                    rho_vu=AvgAggregation(),
                    rho_eu=AvgAggregation())
            core_blocks.append(block)
        self.core_blocks = ListModule(*core_blocks)

        self.dec = GNBlock(phi_u=DecoderGlobalStateUpdate())  # maps global state from vec to scalar

    def forward(self, g: Graph) -> torch.Tensor:

        # add/remove/keep edges
        g = self.edge_fns[self.edge_mode](g)

        g = self.enc(g)
        for core in self.core_blocks:
            g = core(g)
        g: Graph = self.dec(g)

        return g.attr.val

    @staticmethod
    def no_edges(g: Graph) -> Graph:
        g = deepcopy(g)
        g.remove_all_edges()
        g.add_reflexive_edges()
        return g

    @staticmethod
    def all_edges(g: Graph) -> Graph:
        g = deepcopy(g)
        g.remove_all_edges()
        g.add_all_edges(reflexive=True)
        return g

    @staticmethod
    def default(g: Graph) -> Graph:
        g = deepcopy(g)
        g.add_reflexive_edges()
        return g

    @staticmethod
    def bi_directional(g: Graph) -> Graph:
        g = deepcopy(g)

        new_edges = set()
        for e in g.edges:
            new_edges.add(Edge(sender=e.receiver, receiver=e.sender, attr=e.attr))
        g.add_reflexive_edges()
        for e in new_edges:
            g.edges.add(e)

        return g

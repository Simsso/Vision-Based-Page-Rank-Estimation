import torch
from graph_nets.functions.aggregation import AvgAggregation

from graph_nets.block import GNBlock

from graph_nets.data_structures.graph import Graph
from torch import nn, Tensor

from graph_nets.functions.update import NodeAggregationGlobalStateUpdate
from rank_predictor.model.graph_extractor_full import DecoderGlobalStateUpdate
from rank_predictor.model.utils import get_extraction_block


class GNAvg(nn.Module):

    def __init__(self, feat_extr: nn.Module):
        super().__init__()

        self.extr_block = get_extraction_block(feat_extr)

        self.dec1 = GNBlock(
            phi_u=NodeAggregationGlobalStateUpdate(),
            rho_vu=AvgAggregation()
        )
        self.dec2 = GNBlock(
            phi_u=DecoderGlobalStateUpdate()
        )

    def forward(self, g: Graph) -> Graph:
        with torch.no_grad():
            self.extr_block.eval()
            g: Graph = self.extr_block(g)

        g.add_reflexive_edges()  # ensure that average aggregations have at least one value to work with

        g: Graph = self.dec1(g)
        g: Graph = self.dec2(g)

        return g.attr.val

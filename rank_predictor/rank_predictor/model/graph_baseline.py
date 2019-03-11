import torch
from typing import Dict
from torch import nn
from graph_nets import Graph, GNBlock, IdentityEdgeUpdate, IndependentNodeUpdate, Attribute, GlobalStateUpdate, \
    ConstantAggregation, AvgAggregation
from rank_predictor.model.screenshot_feature_extractor import DesktopScreenshotFeatureExtractor


class NodeAggregationGlobalStateUpdate(GlobalStateUpdate):

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return aggr_v


class GraphBaseline(nn.Module):

    def __init__(self):
        super().__init__()

        self.desktop_screenshot_extractor = DesktopScreenshotFeatureExtractor()

        def node_update_fn(node: Dict[str, any]) -> Attribute:
            desktop_img: torch.Tensor = node['desktop_img']

            rank_scalar = self.desktop_screenshot_extractor(desktop_img.unsqueeze(0))

            return rank_scalar

        self.graph_block = GNBlock(
            phi_e=IdentityEdgeUpdate(),
            phi_v=IndependentNodeUpdate(node_update_fn),
            phi_u=NodeAggregationGlobalStateUpdate(),
            rho_ev=ConstantAggregation(),
            rho_vu=AvgAggregation(),
            rho_eu=ConstantAggregation())

    def forward(self, g: Graph) -> Graph:
        g_out: Graph = self.graph_block(g)
        return g_out.attr.val

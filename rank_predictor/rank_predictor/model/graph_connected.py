from typing import Dict
import torch
from graph_nets.data_structures.attribute import Attribute
from rank_predictor.model.util import global_avg_pool
from graph_nets.block import GNBlock
from graph_nets.data_structures.graph import Graph
from graph_nets.functions.aggregation import ConstantAggregation, AvgAggregation
from graph_nets.functions.update import IdentityEdgeUpdate, IndependentNodeUpdate, IdentityGlobalStateUpdate, \
    EdgeUpdate, NodeUpdate, NodeAggregationGlobalStateUpdate
from torch import nn
import torch.nn.functional as F
import rank_predictor.model.util as uf


class GraphConnected(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.pre_process_cnn = PreProcessCNN()

        def first_node_update_fn(node: Dict[str, any]) -> Attribute:
            desktop_img: torch.Tensor = node['desktop_img']

            hidden_layer_out = self.pre_process_cnn(desktop_img.unsqueeze(0))

            return hidden_layer_out

        self.pre_process_block = GNBlock(
            phi_v=IndependentNodeUpdate(first_node_update_fn))

        self.main_block = GNBlock(
            phi_e=SenderGlobalAveragePoolingEdgeUpdate(),
            phi_v=MainNodeUpdate(),
            rho_ev=AvgAggregation())

        self.extraction_block = GNBlock(
            phi_e=SenderGlobalAveragePoolingEdgeUpdate(),
            phi_v=ExtractionNodeUpdate(),
            phi_u=NodeAggregationGlobalStateUpdate(),
            rho_ev=AvgAggregation(),
            rho_vu=AvgAggregation())

    def forward(self, g: Graph) -> Graph:
        g.remove_all_edges()
        g.add_all_edges()
        g: Graph = self.pre_process_block(g)
        g: Graph = self.main_block(g)
        g: Graph = self.extraction_block(g)
        return g.attr.val


class SenderGlobalAveragePoolingEdgeUpdate(EdgeUpdate):

    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        return Attribute(global_avg_pool(v_s.val))


class MainNodeUpdate(NodeUpdate):

    def __init__(self):
        super().__init__()

        self.conv_layers = MainBlockCNN()

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        img_tensor = v.val
        assert aggr_e.val.is_cuda, "Following reshaping does only work on GPU"
        aggr_channels = aggr_e.val.unsqueeze(-1).unsqueeze(-1)  # add height and width dimension for broadcasting

        img_plus_aggr = torch.add(img_tensor, aggr_channels) / 2

        conv_out = self.conv_layers(img_plus_aggr)

        return Attribute(conv_out)


class ExtractionNodeUpdate(NodeUpdate):

    def __init__(self):
        super().__init__()

        self.dense_layers = ExtractionBlockDense()

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        img_tensor = v.val
        assert aggr_e.val.is_cuda, "Following reshaping does only work on GPU"
        aggr_channels = aggr_e.val.unsqueeze(-1).unsqueeze(-1)  # add height and width dimension for broadcasting

        img_plus_aggr = torch.add(img_tensor, aggr_channels) / 2

        conv_out = self.dense_layers(img_plus_aggr)

        return Attribute(conv_out)


class PreProcessCNN(nn.Module):
    """
    First two layer groups of the DesktopScreenshotFeatureExtractor model.
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1a = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv1b = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.conv2a = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.dropout2d(x, p=.25, training=self.training)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, (3, 3))
        x = F.dropout2d(x, p=.25, training=self.training)

        return x


class MainBlockCNN(nn.Module):
    """
    Third and fourth layer groups of the DesktopScreenshotFeatureExtractor model.
    """

    def __init__(self):
        super().__init__()

        self.conv3a = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv3b = nn.Conv2d(128, 128, kernel_size=(3, 3))

        self.conv4a = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.conv4b = nn.Conv2d(256, 256, kernel_size=(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, (3, 3))
        x = F.dropout2d(x, p=.25, training=self.training)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = F.max_pool2d(x, (3, 3))
        x = F.dropout2d(x, p=.25, training=self.training)

        return x


class ExtractionBlockDense(nn.Module):
    """
    Fifth and sixth layer groups
    """

    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = uf.global_avg_pool(x)
        x = uf.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=.3, training=self.training)

        x = self.dense2(x)
        # x = torch.sigmoid(x)
        assert x.size(1) == 1
        x = x.view((-1,))

        return x

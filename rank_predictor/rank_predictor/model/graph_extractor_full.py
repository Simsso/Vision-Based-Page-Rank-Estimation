import torch
from typing import Dict
from graph_nets.functions.aggregation import AvgAggregation
from graph_nets.block import GNBlock
from graph_nets.data_structures.attribute import Attribute
from graph_nets.functions.update import IndependentNodeUpdate, EdgeUpdate, GlobalStateUpdate, NodeUpdate
from graph_nets.data_structures.graph import Graph
from torch import nn, Tensor
import torch.nn.functional as F
import rank_predictor.model.utils as uf


class GraphExtractorFull(nn.Module):

    def __init__(self, num_core_blocks: int, drop_p: float):
        super().__init__()

        self.num_core_blocks = num_core_blocks
        self.drop_p = drop_p

        self.screenshot_feature_extractor = ScreenshotsFeatureExtractor(drop_p)

        def node_extractor_fn(node: Dict[str, any]) -> Tensor:
            desktop_img: Tensor = node['desktop_img']
            mobile_img: Tensor = node['mobile_img']

            # desktop and mobile feature vector
            x1, x2 = self.screenshot_feature_extractor(
                desktop_img.unsqueeze(0),
                mobile_img.unsqueeze(0)
            )

            x = torch.cat((x1, x2), dim=1).view(-1)

            return x

        self.extraction_block = GNBlock(
            phi_v=IndependentNodeUpdate(node_extractor_fn))

        self.encoder = GNBlock(
            phi_e=EncoderEdgeUpdate(),
            phi_u=EncoderGlobalStateUpdate(),
            rho_eu=AvgAggregation()
        )
        self.core = GNBlock(
            phi_e=CoreEdgeUpdate(self.drop_p),
            phi_v=CoreNodeUpdate(self.drop_p),
            phi_u=CoreGlobalStateUpdate(self.drop_p),
            rho_ev=AvgAggregation(),
            rho_vu=AvgAggregation(),
            rho_eu=AvgAggregation()
        )
        self.decoder = GNBlock(
            phi_u=DecoderGlobalStateUpdate())

    def forward(self, g: Graph) -> Graph:
        g: Graph = self.extraction_block(g)

        g.add_reflexive_edges()  # ensure that average aggregations have at least one value to work with
        g: Graph = self.encoder(g)

        for _ in range(self.num_core_blocks):
            g: Graph = self.core(g)

        g: Graph = self.decoder(g)

        return g.attr.val  # global state u


class EncoderEdgeUpdate(EdgeUpdate):
    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        return v_s


class EncoderGlobalStateUpdate(GlobalStateUpdate):
    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return aggr_e


class CoreNodeUpdate(NodeUpdate):
    def __init__(self, drop_p: float):
        super().__init__()
        self.drop_p = drop_p
        self.dense = nn.Linear(64*3, 64)

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:

        x = torch.cat((aggr_e.val, v.val, u.val))
        x = F.relu(self.dense(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)

        return Attribute(x)


class CoreEdgeUpdate(EdgeUpdate):
    def __init__(self, drop_p: float):
        super().__init__()
        self.drop_p = drop_p
        self.dense = nn.Linear(64*4, 64)

    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        x = torch.cat((e.val, v_r.val, v_s.val, u.val))
        x = F.relu(self.dense(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)

        return Attribute(x)


class CoreGlobalStateUpdate(GlobalStateUpdate):
    def __init__(self, drop_p: float):
        super().__init__()
        self.drop_p = drop_p
        self.dense = nn.Linear(64*3, 64)

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        x = torch.cat((aggr_e.val, aggr_v.val, u.val))
        x = F.relu(self.dense(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)

        return Attribute(x)


class DecoderGlobalStateUpdate(GlobalStateUpdate):

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 1)

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return Attribute(self.dense(u.val))


class ScreenshotsFeatureExtractor(nn.Module):
    """
    Converts a screenshot tuple into a feature vector tuple.
    """

    def __init__(self, drop_p: float):
        super().__init__()
        self.drop_p = drop_p

        self.conv1a_d = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv1b_d = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.conv1a_m = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv1b_m = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.conv2a = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 3))

        self.conv3a = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv3b = nn.Conv2d(128, 128, kernel_size=(3, 3))

        self.conv4a = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.conv4b = nn.Conv2d(256, 256, kernel_size=(3, 3))

        self.dense1 = nn.Linear(256, 256)
        self.dense2_d = nn.Linear(256, 32)
        self.dense2_m = nn.Linear(256, 32)

    def forward(self, desktop_img: Tensor, mobile_img: Tensor, return_feature_maps: bool = False) -> (Tensor, Tensor):
        d_feat_maps = {}
        m_feat_maps = {}

        xd = desktop_img
        xm = mobile_img

        d_feat_maps['img'] = xd
        m_feat_maps['img'] = xm

        xd = F.relu(self.conv1a_d(xd))
        xd = self.conv1b_d(xd)
        d_feat_maps['conv1'] = xd
        xd = F.relu(xd)
        xd = F.max_pool2d(xd, (2, 2))
        xd = F.dropout2d(xd, p=self.drop_p, training=self.training)

        xd = F.relu(self.conv2a(xd))
        xd = self.conv2b(xd)
        d_feat_maps['conv2'] = xd
        xd = F.relu(xd)
        xd = F.max_pool2d(xd, (3, 3))
        xd = F.dropout2d(xd, p=self.drop_p, training=self.training)

        xd = F.relu(self.conv3a(xd))
        xd = self.conv3b(xd)
        d_feat_maps['conv3'] = xd
        xd = F.relu(xd)
        xd = F.max_pool2d(xd, (3, 3))
        xd = F.dropout2d(xd, p=self.drop_p, training=self.training)

        xd = F.relu(self.conv4a(xd))
        xd = self.conv4b(xd)
        d_feat_maps['conv4'] = xd
        xd = F.relu(xd)
        xd = F.max_pool2d(xd, (3, 3))
        xd = F.dropout2d(xd, p=self.drop_p, training=self.training)

        xd = uf.global_avg_pool(xd)
        xd = uf.flatten(xd)
        xd = F.relu(self.dense1(xd))
        xd = F.dropout(xd, p=self.drop_p, training=self.training)

        xd = self.dense2_d(xd)

        xm = F.relu(self.conv1a_m(xm))
        xm = self.conv1b_m(xm)
        m_feat_maps['conv1'] = xm
        xm = F.relu(xm)
        xm = F.max_pool2d(xm, (2, 2))
        xm = F.dropout2d(xm, p=self.drop_p, training=self.training)

        xm = F.relu(self.conv2a(xm))
        xm = self.conv2b(xm)
        m_feat_maps['conv2'] = xm
        xm = F.relu(xm)
        xm = F.max_pool2d(xm, (3, 3))
        xm = F.dropout2d(xm, p=self.drop_p, training=self.training)

        xm = F.relu(self.conv3a(xm))
        xm = self.conv3b(xm)
        m_feat_maps['conv3'] = xm
        xm = F.relu(xm)
        xm = F.max_pool2d(xm, (3, 3))
        xm = F.dropout2d(xm, p=self.drop_p, training=self.training)

        xm = F.relu(self.conv4a(xm))
        xm = self.conv4b(xm)
        m_feat_maps['conv4'] = xm
        xm = F.relu(xm)
        xm = F.max_pool2d(xm, (3, 3))
        xm = F.dropout2d(xm, p=self.drop_p, training=self.training)

        xm = uf.global_avg_pool(xm)
        xm = uf.flatten(xm)
        xm = F.relu(self.dense1(xm))
        xm = F.dropout(xm, p=self.drop_p, training=self.training)

        xm = self.dense2_m(xm)

        if return_feature_maps:
            return xd, xm, d_feat_maps, m_feat_maps

        return xd, xm


class ScreenshotsFeatureExtractorWithHead(nn.Module):
    """
    Converts a screenshot tuple into a scalar by adding a linear layer to the `ScreenshotsFeatureExtractor`.
    """

    def __init__(self, drop_p: float):
        super().__init__()

        self.extractor = ScreenshotsFeatureExtractor(drop_p)
        self.head = nn.Linear(64, 1)

    def forward(self, desktop_imgs: Tensor, mobile_imgs: Tensor) -> Tensor:
        """
        :return: Scalar tensor
        """

        x = self.extractor(desktop_imgs, mobile_imgs)
        x = torch.cat(x, dim=1)
        x = F.relu(x)
        x = self.head(x).view(-1)

        return x

    def state_dict(self, **kwargs):
        return {
            'extractor': self.extractor.state_dict(**kwargs),
            'head': self.head.state_dict(**kwargs)
        }

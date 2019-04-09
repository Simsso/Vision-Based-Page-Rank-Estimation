from typing import Callable
from torch import nn
from graph_nets.data_structures.attribute import Attribute


class EdgeUpdate(nn.Module):

    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        """
        Edge update function phi^(e). Converts e into e' given the attributes of the two adjacent nodes and the global
        state u.
        :param e: Edge attribute
        :param v_r: Receiver node attribute
        :param v_s: Sender node attribute
        :param u: Global state
        :return: New edge attribute e'
        """
        raise NotImplementedError


class NodeUpdate(nn.Module):

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        """
        Node update function phi^(v). Converts v into v' given the aggregated, updated edge attributes of all adjacent
        edges and the global state u.
        :param aggr_e: Aggregated, updated local(!) edges
        :param v: Node attribute
        :param u: Global state
        :return: New node attribute v'
        """
        raise NotImplementedError


class GlobalStateUpdate(nn.Module):

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        """
        Global state update function phi^(u). Converts u into u' given the aggregated edges and aggregated nodes.
        :param aggr_e: Aggregated edges
        :param aggr_v: Aggregated nodes
        :param u: Global state
        :return: New global state u'
        """
        raise NotImplementedError


class IndependentEdgeUpdate(EdgeUpdate):
    """
    Independent edge update (i.e. neighbor information is disregarded) with a given function.
    """

    def __init__(self, mapping_fn: Callable[[any], any]) -> None:
        super().__init__()
        self.mapping_fn = mapping_fn

    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        transformed = self.mapping_fn(e.val)
        return Attribute(transformed)


class IndependentNodeUpdate(NodeUpdate):
    """
    Independent node update (i.e. neighbor information is disregarded) with a given function.
    """

    def __init__(self, mapping_fn: Callable[[any], any]) -> None:
        super().__init__()
        self.mapping_fn = mapping_fn

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        transformed = self.mapping_fn(v.val)
        return Attribute(transformed)


class IndependentGlobalStateUpdate(GlobalStateUpdate):
    """
    Independent global state update (i.e. node and edge information is disregarded) with a given function.
    """

    def __init__(self, mapping_fn: Callable[[any], any]) -> None:
        super().__init__()
        self.mapping_fn = mapping_fn

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        transformed = self.mapping_fn(u.val)
        return Attribute(transformed)


class IdentityEdgeUpdate(EdgeUpdate):
    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        return e


class IdentityNodeUpdate(NodeUpdate):
    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        return v


class IdentityGlobalStateUpdate(GlobalStateUpdate):
    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return u


class NodeAggregationGlobalStateUpdate(GlobalStateUpdate):

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return aggr_v

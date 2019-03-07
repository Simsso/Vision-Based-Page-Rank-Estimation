from unittest import TestCase
from copy import deepcopy
from graph_nets import Graph, Node, Attribute, Edge, GNBlock, EdgeUpdate, NodeUpdate, GlobalStateUpdate, \
    ScalarSumAggregation


class SenderIdentityEdgeUpdate(EdgeUpdate):
    """
    Copies the sender node attribute into the edge attribute.
    """

    def forward(self, e: Attribute, v_r: Attribute, v_s: Attribute, u: Attribute) -> Attribute:
        return deepcopy(v_s)


class EdgeNodeSumNodeUpdate(NodeUpdate):
    """
    Adds the aggregated edge values to the node value.
    """

    def forward(self, aggr_e: Attribute, v: Attribute, u: Attribute) -> Attribute:
        return Attribute(aggr_e.val + v.val)


class MixedGlobalStateUpdate(GlobalStateUpdate):
    """
    Adds aggregated nodes and edges to the negative global state.
    """

    def forward(self, aggr_e: Attribute, aggr_v: Attribute, u: Attribute) -> Attribute:
        return Attribute(aggr_e.val + aggr_v.val - u.val)


class TestGNBlock(TestCase):

    def test_basic(self):
        """
        Basic test w/o PyTorch, all attributes are scalars, edges do not have attributes.
        Feeds a graph through a basic graph block twice and compares to the target values after both passes.
        """

        # create data structure
        v_0, v_1, v_2 = Node(Attribute(1)), Node(Attribute(10)), Node(Attribute(20))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1), Edge(v_0, v_2), Edge(v_1, v_2)]
        g_0 = Graph(nodes=vs, edges=es, attr=Attribute(0))

        # create block w/ functions
        block = GNBlock(
            phi_e=SenderIdentityEdgeUpdate(),
            phi_v=EdgeNodeSumNodeUpdate(),
            phi_u=MixedGlobalStateUpdate(),
            rho_ev=ScalarSumAggregation(),
            rho_vu=ScalarSumAggregation(),
            rho_eu=ScalarSumAggregation())

        g_1 = block(g_0)

        v_0, v_1, v_2 = Node(Attribute(1)), Node(Attribute(10+1)), Node(Attribute(20+11))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1, Attribute(1)), Edge(v_0, v_2, Attribute(1)), Edge(v_1, v_2, Attribute(10))]
        g_1_target = Graph(nodes=vs, edges=es, attr=Attribute(35))

        self.assertTrue(g_1 == g_1_target)

        g_2 = block(g_1)

        v_0, v_1, v_2 = Node(Attribute(1)), Node(Attribute(10+2)), Node(Attribute(20+11+12))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1, Attribute(1)), Edge(v_0, v_2, Attribute(1)), Edge(v_1, v_2, Attribute(11))]
        g_2_target = Graph(nodes=vs, edges=es, attr=Attribute(1 + 12 + 43 - 35))

        self.assertTrue(g_2 == g_2_target)

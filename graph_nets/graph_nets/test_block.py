from typing import Tuple
import unittest
from unittest import TestCase
from copy import deepcopy
import torch
from torch import optim
from torch.nn import MSELoss
from graph_nets import Graph, Node, Attribute, Edge, GNBlock, EdgeUpdate, NodeUpdate, GlobalStateUpdate, \
    ScalarSumAggregation, LinearIndependentGNBlock
from graph_nets.functions.loss import GraphLoss


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

        v_0, v_1, v_2 = Node(Attribute(1)), Node(Attribute(10 + 1)), Node(Attribute(20 + 11))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1, Attribute(1)), Edge(v_0, v_2, Attribute(1)), Edge(v_1, v_2, Attribute(10))]
        g_1_target = Graph(nodes=vs, edges=es, attr=Attribute(35))

        self.assertTrue(g_1 == g_1_target)

        g_2 = block(g_1)

        v_0, v_1, v_2 = Node(Attribute(1)), Node(Attribute(10 + 2)), Node(Attribute(20 + 11 + 12))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1, Attribute(1)), Edge(v_0, v_2, Attribute(1)), Edge(v_1, v_2, Attribute(11))]
        g_2_target = Graph(nodes=vs, edges=es, attr=Attribute(1 + 12 + 43 - 35))

        self.assertTrue(g_2 == g_2_target)


code_size = 32


class LinearGNBlock(TestCase):
    """
    Tests a GN block that applies a linear transformation independently.
    """

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(15092017)

    def test_forward_pass(self):
        linear_block = LinearIndependentGNBlock(
            e_config=(4, 8, True),
            v_config=(1, 1, False),
            u_config=(16, 16, True))

        nodes = Node.from_vals([torch.randn(1), torch.randn(1), torch.randn(1)])
        edges = [Edge(nodes[0], nodes[1], Attribute(torch.randn(4))),
                 Edge(nodes[1], nodes[2], Attribute(torch.randn(4))),
                 Edge(nodes[2], nodes[1], Attribute(torch.randn(4)))]
        g_in = Graph(nodes, edges, Attribute(torch.randn(16)))

        # noinspection PyUnusedLocal
        g_out = linear_block(g_in)

        self.assertTrue(True)  # the assertion is that the forward pass works without errors

    def test_learn_identity(self):
        # construct input graph with random values
        nodes = Node.from_vals([torch.randn(1), torch.randn(1), torch.randn(1)])
        edges = [Edge(nodes[0], nodes[1], Attribute(torch.randn(4))),
                 Edge(nodes[1], nodes[2], Attribute(torch.randn(4))),
                 Edge(nodes[2], nodes[1], Attribute(torch.randn(4)))]
        g_in = Graph(nodes, edges, Attribute(torch.randn(16)))
        g_target = deepcopy(g_in)

        block_1 = LinearIndependentGNBlock(e_config=(4, 8, True), v_config=(1, 12, True), u_config=(16, 16, True))
        block_2 = LinearIndependentGNBlock(e_config=(8, 4, False), v_config=(12, 1, False), u_config=(16, 16, False))

        model = torch.nn.Sequential(block_1, block_2)
        opt = optim.SGD(model.parameters(), lr=.1, momentum=0)
        loss_fn = GraphLoss(e_fn=MSELoss(), v_fn=MSELoss(), u_fn=MSELoss())

        loss = torch.Tensor([1.])
        for step in range(100):
            model.train()
            opt.zero_grad()
            g_out = model(g_in)
            loss = loss_fn(g_out, g_target)
            loss.backward()
            opt.step()

        final_loss = loss.detach().numpy()
        print(final_loss)
        self.assertTrue(final_loss < 1e-3)


class TestGNBlockModuleSortDemo(TestCase):
    """
    Tests of the PyTorch interface for the graph network block.
    """

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(15092017)

    @staticmethod
    def _create_tuple(x_min: int, x_max: int) -> Tuple[Graph, Graph]:
        """
        Creates a sorting task training sample, i.e. a tuple consisting of (input, target).
        :param x_min: Number min value
        :param x_max: Number max value (exclusive)
        :return: Two graphs, both fully connected, one with randomly initialized attributes, the other one with targets.
        """
        assert x_min < x_max

        vec = torch.arange(x_min, x_max)
        nodes = [Node(Attribute(x)) for x in vec]

        g_in = Graph(nodes, [])
        g_out = deepcopy(g_in)

        def sorted_edge_attribute_generator(n1: Node, n2: Node) -> Attribute:
            val = [0, 1] if n1.attr.val < n2.attr.val else [1, 0]
            return Attribute(torch.Tensor(val).float())

        g_in.add_all_edges(reflexive=False, attribute_generator=lambda n1, n2: Attribute(torch.randn(2)))
        g_out.add_all_edges(reflexive=False, attribute_generator=sorted_edge_attribute_generator)

        return g_in, g_out

    @staticmethod
    def _build_encoder() -> GNBlock:
        return LinearIndependentGNBlock(e_config=(2, code_size, True), v_config=None, u_config=None)

    @staticmethod
    def _build_core_block() -> GNBlock:
        raise NotImplementedError()

    @staticmethod
    def _build_decoder() -> GNBlock:
        return LinearIndependentGNBlock(e_config=(code_size, 2, True), v_config=None, u_config=None)

    @unittest.skip("Not implemented yet")
    def test_sort_demo(self):
        """
        Sort demo, inspired by this example (https://github.com/deepmind/graph_nets#run-sort-demo-in-browser).
        Short description: Represent a list of elements as a graph, where each number corresponds to a node. The graph
        is fully connected and the training target is to learn to mark the starting node and edges that connect from
        smaller to next larger element (similar to a linked list).
        """
        raise NotImplementedError()

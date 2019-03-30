from copy import deepcopy
from unittest import TestCase
import numpy as np
from torch import Tensor
from torch.nn import MSELoss, L1Loss
from graph_nets.data_structures.attribute import Attribute
from graph_nets.data_structures.edge import Edge
from graph_nets.data_structures.graph import Graph
from graph_nets.data_structures.node import Node
from graph_nets.functions.loss import GraphLoss


class TestGraphLoss(TestCase):
    def test_scalar_edge_loss(self):
        nodes = [Node(), Node(), Node()]
        edges = [Edge(nodes[0], nodes[1], Attribute(Tensor([-50.]))),
                 Edge(nodes[1], nodes[0], Attribute(Tensor([-40.]))),
                 Edge(nodes[0], nodes[2], Attribute(Tensor([1.])))]
        g1 = Graph(nodes, edges)

        g2 = deepcopy(g1)
        g2.ordered_edges[0].attr.val = Tensor([-45.])
        g2.ordered_edges[1].attr.val = Tensor([-40.])
        g2.ordered_edges[2].attr.val = Tensor([1.1])

        loss = GraphLoss(e_fn=MSELoss())
        loss_val = loss(g1, g2).detach().numpy()
        target_loss_val = ((-50.+45.)**2 + (-40.+40.)**2 + (1-1.1)**2) / 3

        self.assertTrue(np.isclose(loss_val, target_loss_val))

    def test_vector_edge_loss(self):
        nodes = [Node(), Node(), Node()]
        edges = [Edge(nodes[0], nodes[1], Attribute(Tensor([-50., -10., -5.]))),
                 Edge(nodes[1], nodes[0], Attribute(Tensor([-40., 100., 120.]))),
                 Edge(nodes[0], nodes[2], Attribute(Tensor([1., 3., 4.]))),
                 Edge(nodes[0], nodes[0], Attribute(Tensor([2., 2., 2.])))]
        g1 = Graph(nodes, edges)

        g2 = deepcopy(g1)
        g2.ordered_edges[0].attr.val = Tensor([-45., -11., -5.])
        g2.ordered_edges[1].attr.val = Tensor([-40., 200., 121.])
        g2.ordered_edges[2].attr.val = Tensor([1.1, 3., 3.9])
        g2.ordered_edges[3].attr.val = Tensor([2., 2., 2.1])

        loss = GraphLoss(e_fn=MSELoss())
        loss_val = loss(g1, g2).detach().numpy()

        # division by 3 because there are three entries per vector
        # division by 4 because there are four edges
        target_loss_val = ((-50.+45.)**2 + (-10.+11.)**2 + (-40.+40.)**2 + (100.-200.)**2 + (120.-121.)**2 +
                           (1-1.1)**2 + (4.-3.9)**2 + (2-2.1)**2) / 3 / 4

        self.assertTrue(np.isclose(loss_val, target_loss_val))

    def test_combined_loss(self):
        nodes = [Node(Attribute(Tensor([-4., -8.]))),
                 Node(Attribute(Tensor([1., 5.]))),
                 Node(Attribute(Tensor([4., 4.]))),
                 Node(Attribute(Tensor([0., 1., 5.])))]
        edges = [Edge(nodes[0], nodes[1], Attribute(Tensor([1., 2., 3.]))),
                 Edge(nodes[1], nodes[2], Attribute(Tensor([1., 2.]))),
                 Edge(nodes[2], nodes[1], Attribute(Tensor([5.]))),
                 Edge(nodes[1], nodes[3], Attribute(Tensor([1., 2., 3., 4.])))]
        u = Attribute(Tensor([[1., 2., 4., 3.], [8., 3., 0., 3.], [1., 7., 5., 3.]]))
        g1 = Graph(nodes, edges, attr=u)

        g2 = deepcopy(g1)
        g2.ordered_nodes[0].attr.val = Tensor([-4., -8.1])
        g2.ordered_nodes[1].attr.val = Tensor([2., 6.])
        g2.ordered_nodes[3].attr.val = Tensor([1., 1.5, 5.])
        g2.ordered_edges[0].attr.val = Tensor([2., 3., 4.])
        g2.ordered_edges[1].attr.val = Tensor([5., 10.])
        g2.attr.val = Tensor([[2., 2., 4., 3.], [100, 3., 1., 3.], [1., 14., 5., 3.]])

        loss = GraphLoss(e_fn=MSELoss(), v_fn=L1Loss(), u_fn=MSELoss())
        loss_val = loss(g1, g2).detach().numpy()
        e_loss = (1. + (4**2 + 8**2)/2) / 4
        v_loss = (.1/2 + 2./2 + (1 + .5)/3) / 4
        u_loss = (1 + (8-100)**2 + 1 + 7**2) / 12 / 1
        target_loss_val = v_loss + e_loss + u_loss

        self.assertTrue(np.isclose(loss_val, target_loss_val))

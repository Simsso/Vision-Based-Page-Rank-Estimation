from unittest import TestCase

from graph_nets.data_structures.attribute import Attribute
from graph_nets.data_structures.edge import Edge
from graph_nets.data_structures.graph import Graph
from graph_nets.data_structures.node import Node


class TestGraph(TestCase):

    def test_integrity_check(self):
        """
        Create one valid and one invalid graph, the invalid graph has an edge that points to a node which is not
        contained in the set of nodes.
        """
        v_0, v_1, v_2 = Node(), Node(), Node()
        v_3 = Node()
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1), Edge(v_0, v_2)]

        Graph(nodes=vs, edges=es)  # pass for valid edge set

        es.append(Edge(v_1, v_3))
        with self.assertRaises(ValueError):
            Graph(nodes=vs, edges=es)

    def test_graph_equality(self):
        v_0, v_1, v_2 = Node(Attribute(0.)), Node(Attribute(1.)), Node(Attribute(2.))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1), Edge(v_0, v_2)]  # edges
        g_0 = Graph(nodes=vs, edges=es)

        v_0, v_1, v_2 = Node(Attribute(0.)), Node(Attribute(1.)), Node(Attribute(2.))
        vs = [v_0, v_1, v_2]  # nodes
        es = [Edge(v_0, v_1), Edge(v_0, v_2)]  # edges
        g_1 = Graph(nodes=vs, edges=es)

        self.assertTrue(g_0 == g_1)

    def test_input_dtype_check(self):
        with self.assertRaises(AssertionError):
            # noinspection PyTypeChecker
            Graph({Node()}, edges=[])

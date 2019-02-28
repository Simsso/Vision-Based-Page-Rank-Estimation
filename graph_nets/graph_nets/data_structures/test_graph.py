from unittest import TestCase

from graph_nets import Node, Attribute, Graph, Edge


class TestGraph(TestCase):

    def test_integrity_check(self):
        """
        Create one valid and one invalid graph, the invalid graph has an edge that points to a node which is not
        contained in the set of nodes.
        """
        v_0, v_1, v_2 = Node(), Node(), Node()
        v_3 = Node()
        vs = {v_0, v_1, v_2}  # nodes
        es = {Edge(v_0, v_1), Edge(v_0, v_2)}

        Graph(nodes=vs, edges=es)  # pass for valid edge set

        es.add(Edge(v_1, v_3))
        with self.assertRaises(AssertionError):
            Graph(nodes=vs, edges=es)

    def test_graph_equality(self):
        v_0, v_1, v_2 = Node(Attribute(0.)), Node(Attribute(1.)), Node(Attribute(2.))
        vs = {v_0, v_1, v_2}  # nodes
        es = {Edge(v_0, v_1), Edge(v_0, v_2)}  # edges
        g_0 = Graph(nodes=vs, edges=es)

        v_0, v_1, v_2 = Node(Attribute(0.)), Node(Attribute(1.)), Node(Attribute(2.))
        vs = {v_1, v_2, v_0}  # nodes
        es = {Edge(v_0, v_2), Edge(v_0, v_1)}  # edges
        g_1 = Graph(nodes=vs, edges=es)

        self.assertTrue(g_0 == g_1)

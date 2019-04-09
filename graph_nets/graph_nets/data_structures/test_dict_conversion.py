from unittest import TestCase
from graph_nets.data_structures.attribute import Attribute
from graph_nets.data_structures.edge import Edge
from graph_nets.data_structures.graph import Graph
from graph_nets.data_structures.node import Node


class TestDictConversion(TestCase):

    def test_identity_property(self):
        nodes = [Node(Attribute([1, 23, 4])), Node(Attribute("stringattr")), Node(Attribute([1])), Node(Attribute(5))]
        edges = [Edge(nodes[0], nodes[1], Attribute({'dict': 1234})),
                 Edge(nodes[0], nodes[1], Attribute([1, 2, 3])),
                 Edge(nodes[0], nodes[2], Attribute(5)),
                 Edge(nodes[1], nodes[2], Attribute([3, 4, 5])),
                 Edge(nodes[1], nodes[1], Attribute())]
        global_state = Attribute([[1, 2, 3], [5, 6, 7]])
        g1 = Graph(nodes, edges, global_state)

        g1_prime = Graph.from_dict(g1.asdict())

        self.assertTrue(g1 == g1_prime)

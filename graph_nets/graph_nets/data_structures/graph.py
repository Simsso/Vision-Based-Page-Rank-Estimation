from typing import Optional, Callable, List
from graph_nets.data_structures.utils import lists_equal
from . import Attribute
from graph_nets.data_structures.node import Node
from . import Edge


class Graph:
    """
    Following the DeepMind "Graph Nets" paper (https://arxiv.org/abs/1806.01261).
    This graph is a directed, attributed multi-graph with a global attribute.
    Directed, because edges distinguish between sender and receiver. An attribute can be 'any' information. Multi-graph,
    because there can be more than one edge between vertices, including self-edges.
    """

    def __init__(self, nodes: List[Node], edges: List[Edge], attr: Optional[Attribute] = None):
        if attr is None:
            attr = Attribute(val=None)

        assert isinstance(nodes, list) and isinstance(edges, list), "Nodes and edges must be lists"

        self.nodes = set(nodes)  # V
        self.edges = set(edges)  # E
        self.attr = attr  # e

        # store ordered representations for easy alignment of two graphs with equal structure
        self.ordered_nodes = nodes
        self.ordered_edges = edges

        self._check_integrity()

    def _check_integrity(self) -> None:
        self._check_node_edge_integrity()
        self._check_ordered_references_integrity()

    def _check_node_edge_integrity(self) -> None:
        for edge in self.edges:
            if edge.receiver not in self.nodes or edge.sender not in self.nodes:
                raise ValueError("Edges must only connect nodes that are contained in the graph")

    def _check_ordered_references_integrity(self) -> None:
        if len(self.ordered_edges) != len(self.edges):
            raise ValueError("Number of ordered edges must be equal to the number of edges in the hash set")

        if len(self.ordered_nodes) != len(self.nodes):
            raise ValueError("Number of ordered nodes must be equal to the number of nodes in the hash set")

        for e in self.ordered_edges:
            if e not in self.edges:
                raise ValueError("Every edge in edges_ordered must be contained in edges")

        for n in self.ordered_nodes:
            if n not in self.nodes:
                raise ValueError("Every node in nodes_ordered must be contained in nodes")

    def add_node(self, new_node: Node) -> None:
        self.nodes.add(new_node)
        self.ordered_nodes.append(new_node)
        self._check_integrity()

    def add_edge(self, new_edge: Edge) -> None:
        self.edges.add(new_edge)
        self.ordered_edges.append(new_edge)
        self._check_integrity()

    def add_all_edges(self, reflexive: bool = True, attribute_generator: Optional[Callable[[Node, Node], Attribute]] = None) -> None:
        """
        Modifies the graph in-place, such that it is fully connected, adding n^n edges, where n is the number of nodes.
        :param reflexive: Whether to connect nodes to themselves
        :param attribute_generator: New edges will be given an attribute generated by the attribute generator
        """
        if attribute_generator is None:
            attribute_generator = lambda sn, rn: Attribute()

        for n1 in self.nodes:
            for n2 in self.nodes:
                if not reflexive and n1 == n2:
                    continue
                e = Edge(n1, n2, attribute_generator(n1, n2))
                self.add_edge(e)

    def __eq__(self, g2: object) -> bool:
        if not isinstance(g2, Graph):
            return False

        g1 = self
        g1._check_integrity()
        g2._check_integrity()

        if not lists_equal(g1.ordered_nodes, g2.ordered_nodes, comparator=Node.eq_attr):
            return False

        if not lists_equal(g1.ordered_edges, g2.ordered_edges, comparator=Edge.eq_attr_and_ctx):
            return False

        return True

    def __repr__(self) -> str:
        return self.attr.__repr__()



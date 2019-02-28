from typing import Set, Optional

from graph_nets.data_structures.utils import sets_equal
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

    def __init__(self, nodes: Set[Node], edges: Set[Edge], attribute: Optional[Attribute] = None):
        if attribute is None:
            attribute = Attribute(val=None)

        self.nodes = nodes  # V
        self.edges = edges  # E
        self.attribute = attribute  # e

        assert self._check_node_edge_integrity(), "Edges must only connect nodes that are contained in the graph"

    def _check_node_edge_integrity(self) -> bool:
        for edge in self.edges:
            if edge.receiver not in self.nodes or edge.sender not in self.nodes:
                return False
        return True

    def add_node(self, node: Node) -> None:
        self.nodes.add(node)

    def __eq__(self, g2: object) -> bool:
        """
        This function does not work yet. In some cases it may return True for graphs which are not equal or vice-versa.
        # TODO: fix special cases
        """
        g1 = self

        if not isinstance(g2, Graph):
            return False

        if not sets_equal(g1.nodes, g2.nodes, comparator=Node.eq_attr):
            return False

        if not sets_equal(g1.edges, g2.edges, comparator=Edge.eq_attr_and_ctx):
            return False

        return True

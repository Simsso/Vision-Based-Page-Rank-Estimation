from typing import Set

from rank_predictor.data_structures.attribute import Attribute
from rank_predictor.data_structures.edge import Edge
from rank_predictor.data_structures.node import Node


class Graph:
    """
    Following the DeepMind "Graph Nets" paper (https://arxiv.org/abs/1806.01261).
    This graph is a directed, attributed multi-graph with a global attribute.
    Directed, because edges distinguish between sender and receiver. An attribute can be 'any' information. Multi-graph,
    because there can be more than one edge between vertices, including self-edges.
    """

    def __init__(self, nodes: Set[Node], edges: Set[Edge], attribute: Attribute):
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


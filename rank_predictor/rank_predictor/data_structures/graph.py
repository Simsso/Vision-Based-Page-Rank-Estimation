from typing import Set

from data_structures.attribute import Attribute
from data_structures.edge import Edge
from data_structures.node import Node


class Graph:

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

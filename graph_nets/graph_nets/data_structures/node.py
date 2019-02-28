from typing import Set, Optional
from .attribute import Attribute


class Node:
    """
    A node in a graph. It has an attribute that can hold any value.
    """

    def __init__(self, attribute: Optional[Attribute] = None) -> None:
        if attribute is None:
            attribute = Attribute(val=None)

        self.attribute = attribute  # v
        from .edge import Edge
        self.receiving_edges: Set[Edge] = set()  # incoming
        self.sending_edges: Set[Edge] = set()  # outgoing

    @staticmethod
    def eq_attr(n1: 'Node', n2: 'Node') -> bool:
        """
        Equality check for two nodes. Does not compare receiving and sending edges, i.e. it disregards the context.
        """
        return n1.attribute == n2.attribute



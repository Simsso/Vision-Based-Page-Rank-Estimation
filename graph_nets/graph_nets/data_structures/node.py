from typing import Set, Optional, List
from .attribute import Attribute


class Node:
    """
    A node in a graph. It has an attribute that can hold any value.
    """

    def __init__(self, attr: Optional[Attribute] = None) -> None:
        if attr is None:
            attr = Attribute(val=None)

        self.attr = attr  # v
        from .edge import Edge
        self.receiving_edges: Set[Edge] = set()  # incoming
        self.sending_edges: Set[Edge] = set()  # outgoing

    def __repr__(self) -> str:
        return self.attr.__repr__()

    @staticmethod
    def from_vals(vals: List[any]) -> List['Node']:
        return [Node(Attribute(x)) for x in vals]

    @staticmethod
    def eq_attr(n1: 'Node', n2: 'Node') -> bool:
        """
        Equality check for two nodes. Does not compare receiving and sending edges, i.e. it disregards the context.
        """
        return n1.attr == n2.attr



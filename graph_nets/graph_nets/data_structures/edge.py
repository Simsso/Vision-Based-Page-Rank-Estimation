from typing import Optional
from .attribute import Attribute

from .node import Node


class Edge:
    """
    A directed edge, that connected sender and receiver. The attribute can be any value.
    """

    def __init__(self, sender: Node, receiver: Node, attr: Optional[Attribute] = None) -> None:
        if attr is None:
            attr = Attribute(val=None)

        self.attr: Attribute = attr  # e
        self.sender: Node = sender  # s
        self.receiver: Node = receiver  # r

        sender.sending_edges.add(self)
        receiver.receiving_edges.add(self)

    def __repr__(self) -> str:
        return self.attr.__repr__()

    @staticmethod
    def eq_attr_and_ctx(e1: 'Edge', e2: 'Edge') -> bool:
        """
        Compares two edges, i.e. content and adjacent nodes.
        """
        return e1.attr == e2.attr and \
               Node.eq_attr(e1.sender, e2.sender) and \
               Node.eq_attr(e1.receiver, e2.receiver)

from typing import Optional
from .attribute import Attribute

from .node import Node


class Edge:
    """
    A directed edge, that connected sender and receiver. The attribute can be any value.
    """

    def __init__(self, sender: Node, receiver: Node, attribute: Optional[Attribute] = None) -> None:
        if attribute is None:
            attribute = Attribute(val=None)

        self.attribute = attribute  # e
        self.sender = sender  # s
        self.receiver = receiver  # r

        sender.sending_edges.add(self)
        receiver.receiving_edges.add(self)

    @staticmethod
    def eq_attr_and_ctx(e1: 'Edge', e2: 'Edge') -> bool:
        """
        Compares two edges, i.e. content and adjacent nodes.
        """
        return e1.attribute == e2.attribute and \
               Node.eq_attr(e1.sender, e2.sender) and \
               Node.eq_attr(e1.receiver, e2.receiver)

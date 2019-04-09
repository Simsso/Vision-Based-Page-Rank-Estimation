import torch
from typing import Optional, Dict
from .attribute import Attribute
from .node import Node


class Edge:
    """
    A directed edge, that connects sender and receiver. The attribute can be any value.
    """

    def __init__(self, sender: Node, receiver: Node, attr: Optional[Attribute] = None) -> None:
        """
        :param sender: Sender node of this edge (outgoing)
        :param receiver: Receiver node of this edge (incoming / target)
        :param attr: Optional edge attribute
        """

        if attr is None:
            attr = Attribute(val=None)

        self.attr: Attribute = attr  # e
        self.sender: Node = sender  # s
        self.receiver: Node = receiver  # r

        sender.sending_edges.add(self)
        receiver.receiving_edges.add(self)

    def to(self, device: torch.device) -> 'Edge':
        """
        Moves torch values of this object to the specified device (e.g. GPU).
        """
        self.attr.to(device)
        return self

    def asdict(self) -> Dict:
        return {
            'attr': self.attr.asdict(),
            'sender_hash': hash(self.sender),
            'receiver_hash': hash(self.receiver)
        }

    @staticmethod
    def from_dict(d: Dict, nodes: Dict[str, Node]) -> 'Edge':
        return Edge(
            sender=nodes[d['sender_hash']],
            receiver=nodes[d['receiver_hash']],
            attr=Attribute.from_dict(d['attr'])
        )

    def __repr__(self) -> str:
        return "edge({})".format(self.attr.__repr__())

    @staticmethod
    def eq_attr_and_ctx(e1: 'Edge', e2: 'Edge') -> bool:
        """
        Compares two edges, i.e. content and adjacent nodes (value-wise).
        """
        return (e1.attr == e2.attr and
                Node.eq_attr(e1.sender, e2.sender) and
                Node.eq_attr(e1.receiver, e2.receiver))

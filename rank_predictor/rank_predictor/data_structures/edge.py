from data_structures.attribute import Attribute
from data_structures.node import Node


class Edge:

    def __init__(self, attribute: Attribute, sender: Node, receiver: Node) -> None:
        self.attribute = attribute  # e
        self.sender = sender  # s
        self.receiver = receiver  # r

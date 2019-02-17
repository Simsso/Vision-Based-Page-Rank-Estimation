from data_structures.attribute import Attribute


class Node:
    """
    A node in a graph. It has an attribute that can hold any value.
    """

    def __init__(self, attribute: Attribute) -> None:
        self.attribute = attribute  # v

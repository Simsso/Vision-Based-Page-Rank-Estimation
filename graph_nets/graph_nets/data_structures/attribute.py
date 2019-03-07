from copy import deepcopy
import torch


class Attribute:
    """
    Generic attribute class. An attribute can hold any value and can belong to e.g. graphs, nodes, or edges.
    """

    def __init__(self, val: any = None) -> None:
        self.val = val

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Attribute):
            return True
        return self.val == o.val

    def __repr__(self) -> str:
        return "attr({})".format(self.val.__repr__())

    def __deepcopy__(self, memodict={}):
        """
        The deep copy method is overwritten because PyTorch Tensor attributes cannot be copied automatically by
        deepcopy. Therefore, once we encounter one, we call clone on it.
        """
        if isinstance(self.val, torch.Tensor):
            return Attribute(self.val.clone())
        return Attribute(deepcopy(self.val))

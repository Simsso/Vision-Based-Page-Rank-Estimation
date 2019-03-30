from copy import deepcopy
from typing import Dict
import torch


class Attribute:
    """
    Generic attribute class. An attribute can hold any value and can belong to e.g. graphs, nodes, or edges.
    """

    def __init__(self, val: any = None) -> None:
        self.val = val

    def to(self, device: torch.device) -> 'Attribute':
        """
        Moves torch values of this object to the specified device (e.g. GPU).
        """
        if isinstance(self.val, dict):
            for key, val in self.val.items():
                if isinstance(val, torch.Tensor):
                    self.val[key] = val.to(device)
        elif isinstance(self.val, torch.Tensor):
            self.val = self.val.to(device)
        return self

    def asdict(self) -> Dict:
        if self.val == 'None':
            raise ValueError("Attribute values must not be 'None', however, None is allowed.")
        return {
            'val': self.val if self.val is not None else 'None'
        }

    @staticmethod
    def from_dict(d: Dict) -> 'Attribute':
        val = d['val'] if d['val'] is not 'None' else None
        return Attribute(val)

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

from copy import deepcopy
import torch


class Attribute:

    def __init__(self, val: any = None) -> None:
        self.val = val

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Attribute):
            return True
        return self.val == o.val

    def __repr__(self) -> str:
        return self.val.__repr__()

    def __deepcopy__(self, memodict={}):
        if isinstance(self.val, torch.Tensor):
            return Attribute(self.val.clone())
        return Attribute(deepcopy(self.val))

import torch
from typing import Callable, Optional, List, Union
from torch import Tensor
from torch.nn import MSELoss
from graph_nets import Graph, Attribute

LossFn = Union[Callable[[Tensor, Tensor], Tensor], MSELoss]


class GraphLossPyTorch:
    def __init__(self, e_fn: Optional[LossFn] = None, v_fn: Optional[LossFn] = None, u_fn: Optional[LossFn] = None):
        def id_or_default(fn: Optional[LossFn]) -> LossFn:
            if fn is None:
                return lambda a, b: torch.Tensor([0.])
            return fn

        self.e_fn = id_or_default(e_fn)
        self.v_fn = id_or_default(v_fn)
        self.u_fn = id_or_default(u_fn)

    @staticmethod
    def _compute_loss(loss_fn: LossFn, pred_list: List[Attribute], target_list: List[Attribute]) -> Tensor:
        assert len(pred_list) == len(target_list)

        losses = []
        for pred, target in zip(pred_list, target_list):
            loss = loss_fn(pred.val, target.val)
            assert len(loss.shape) == 0, "Loss function output must be a zero dimensional tensor"
            losses.append(loss.view(1))

        n = len(losses)
        if n == 0:
            return torch.Tensor([0.])

        total_loss = torch.sum(torch.cat(losses)) / n

        return total_loss

    def __call__(self, pred: Graph, target: Graph) -> Tensor:
        e_loss = self._compute_loss(self.e_fn,
                                    [e.attr for e in pred.ordered_edges], [e.attr for e in target.ordered_edges])
        v_loss = self._compute_loss(self.v_fn,
                                    [v.attr for v in pred.ordered_nodes], [v.attr for v in target.ordered_nodes])
        u_loss = self.u_fn(pred.attr.val, target.attr.val)

        return e_loss + v_loss + u_loss

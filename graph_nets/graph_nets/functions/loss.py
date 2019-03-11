import torch
from typing import Callable, Optional, List, Union
from torch import Tensor
from torch.nn import MSELoss

from graph_nets.data_structures.attribute import Attribute
from graph_nets.data_structures.graph import Graph
from graph_nets.utils import tensors_stackable

LossFn = Union[Callable[[Tensor, Tensor], Tensor], MSELoss]


class GraphLoss:
    """
    Given two graphs, this class can compute a loss between them.
    """

    def __init__(self, e_fn: Optional[LossFn] = None, v_fn: Optional[LossFn] = None, u_fn: Optional[LossFn] = None):
        """
        The loss functions may be None, so the corresponding part of the loss is 0. At least one loss function must be
        set. The loss function must support batches.
        :param e_fn: Edge loss function, takes the values of two edges and computes a loss
        :param v_fn: Node loss function, takes the values of two nodes and computes a loss
        :param u_fn: Global state loss function, takes the values of two graph's global states and computes a loss
        """

        assert e_fn is not None or v_fn is not None or u_fn is not None, "At least one loss function must be set"

        self.e_fn, self.v_fn, self.u_fn = e_fn, v_fn, u_fn

    @staticmethod
    def _compute_loss(loss_fn: LossFn, pred_list: List[Attribute], target_list: List[Attribute]) -> Tensor:
        """
        Computes the loss between two lists of attributes using the given loss function. The summed up losses for each
        pair of entries are being averaged to ensure that the loss magnitude does not grow with the size of the graph.
        :param loss_fn: Loss function that takes two values and computes a loss
        :param pred_list: List of predictions (from graph 1)
        :param target_list: List of targets (from graph 2)
        :return: Scalar loss tensor, averaged
        """
        if loss_fn is None:
            return torch.Tensor([0.])

        assert len(pred_list) == len(target_list), \
            "Number of prediction and target values mismatch, got {} and {}.".format(len(pred_list), len(target_list))

        assert len(pred_list) > 0, "Loss cannot be computed on 0 tensors"

        # extract values from attributes
        pred_list = [x.val for x in pred_list]
        target_list = [x.val for x in target_list]

        if tensors_stackable(pred_list):
            # process tensors as a batch
            pred = torch.stack(pred_list)
            target = torch.stack(target_list)

            return loss_fn(pred, target)
        else:
            losses = []
            for pred, target in zip(pred_list, target_list):
                # add batch dimension to both tensors (b=1)
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

                loss = loss_fn(pred, target)

                assert len(loss.shape) == 0, "Loss function output must be a zero dimensional tensor"

                losses.append(loss.view(1))

            n = len(losses)
            total_loss = torch.sum(torch.cat(losses)) / n

            return total_loss

    def __call__(self, pred: Graph, target: Graph) -> Tensor:
        e_loss = self._compute_loss(self.e_fn,
                                    [e.attr for e in pred.ordered_edges], [e.attr for e in target.ordered_edges])
        v_loss = self._compute_loss(self.v_fn,
                                    [v.attr for v in pred.ordered_nodes], [v.attr for v in target.ordered_nodes])
        u_loss = self._compute_loss(self.u_fn, [pred.attr], [target.attr])

        return e_loss + v_loss + u_loss

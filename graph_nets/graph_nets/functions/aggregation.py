import torch
from typing import List, Optional
from torch import nn
from graph_nets.data_structures.attribute import Attribute


class Aggregation(nn.Module):
    """
    While evaluating a graph network block, there are several occasions during which a variably size set must be
    converted into a fixed size representation. For instance when computing a node update, the information from all
    adjacent edges must be considered. There may be 0 to n edges. An aggregation converts a list of attributes into a
    single attribute.
    """

    def forward(self, attrs: List[Attribute]) -> Attribute:
        """
        Edge or node aggregation function rho^(e->v), rho^(v->u), or rho^(e->u). Given a list of edge or node attributes
        it computes an aggregated version which is also an attribute.
        :param attrs: List of edge attributes
        :return: Aggregated attributes
        """
        raise NotImplementedError


class SumAggregation(Aggregation):
    """
    Aggregates the attributes by element-wise summation.
    For instance given the attributes a = [a1 a2 a3] and b = [b1 b2 b3] the aggregation would be [a1+b1 a2+b2 a3+b3].
    """

    def forward(self, attrs: List[Attribute]) -> Attribute:
        assert len(attrs) > 0, "Sum aggregation cannot be applied to empty lists"

        in_shape = attrs[0].val.shape

        attr_vals = [a.val.view(-1) for a in attrs]
        attr_vals_concat = torch.stack(attr_vals)

        attr_vals_sum = torch.sum(attr_vals_concat, dim=0, keepdim=False)

        assert attr_vals_sum.shape == in_shape

        return Attribute(attr_vals_sum)


class AvgAggregation(Aggregation):
    """
    Aggregates the attributes by element-wise average computation.
    For instance given the attributes a = [a1 a2 a3] and b = [b1 b2 b3] the aggregation would be
    [(a1+b1)/2 (a2+b2)/2 (a3+b3)/2].
    """

    def forward(self, attrs: List[Attribute]) -> Attribute:
        n = len(attrs)
        assert n > 0, "Average aggregation cannot be applied to empty lists."

        attr_vals = [a.val for a in attrs]
        attr_vals = torch.stack(attr_vals)

        attr_vals_sum = torch.sum(attr_vals, dim=0, keepdim=False)
        attr_vals_avg = attr_vals_sum / n

        return Attribute(attr_vals_avg)


class ScalarSumAggregation(Aggregation):
    """
    While the SumAggregation class works with PyTorch tensors, this class works with scalar attributes.
    A scalar attribute is e.g. Attribute(5).
    """

    def forward(self, attrs: List[Attribute]) -> Attribute:
        return Attribute(sum([a.val for a in attrs]))


class ConstantAggregation(Aggregation):
    """
    The constant aggregation ignores its inputs and returns a constant value, specified through the constructor.
    It may be helpful in cases where the update of edges / nodes / global state is independent, i.e. neighborhood
    information is discarded anyways.
    """

    def __init__(self, const_val: Optional[any] = None) -> None:
        super().__init__()

        self.constant_val = const_val

    def forward(self, attrs: List[Attribute]) -> Attribute:
        return Attribute(self.constant_val)

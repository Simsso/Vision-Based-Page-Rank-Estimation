import torch
from typing import List, Optional
from graph_nets import Attribute


class Aggregation:
    def __call__(self, attrs: List[Attribute]) -> Attribute:
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

    def __call__(self, attrs: List[Attribute]) -> Attribute:
        attr_vals = [a.val for a in attrs]
        attr_vals = torch.cat(attr_vals)

        attr_vals_sum = torch.sum(attr_vals, dim=0, keepdim=False)

        return Attribute(attr_vals_sum)


class AverageAggregation(Aggregation):
    """
    Aggregates the attributes by element-wise average computation.
    For instance given the attributes a = [a1 a2 a3] and b = [b1 b2 b3] the aggregation would be
    [(a1+b1)/2 (a2+b2)/2 (a3+b3)/2].
    """

    def __call__(self, attrs: List[Attribute]) -> Attribute:
        n = len(attrs)
        assert n > 0, "Average aggregation cannot be applied to empty lists."

        attr_vals = [a.val for a in attrs]
        attr_vals = torch.cat(attr_vals)

        attr_vals_sum = torch.sum(attr_vals, dim=0, keepdim=False)
        attr_vals_avg = attr_vals_sum / n

        return Attribute(attr_vals_avg)


class ScalarSumAggregation(Aggregation):
    def __call__(self, attrs: List[Attribute]) -> Attribute:
        return Attribute(sum([a.val for a in attrs]))


class ConstantAggregation(Aggregation):

    def __init__(self, const_val: Optional[any] = None) -> None:
        super().__init__()

        self.constant_val = const_val

    def __call__(self, attrs: List[Attribute]) -> Attribute:
        return Attribute(self.constant_val)

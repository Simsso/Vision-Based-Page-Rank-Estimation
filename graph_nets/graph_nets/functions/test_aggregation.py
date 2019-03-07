import numpy as np
from typing import List
from unittest import TestCase
import torch
from graph_nets import SumAggregation, Attribute, Aggregation, AvgAggregation


def _feed(aggr: Aggregation, in_val: List[List[float]], out_target: List[float]) -> bool:
    in_attrs = [Attribute(torch.Tensor(np.array(x))) for x in in_val]
    out_target = np.array(out_target)

    out_tensor: torch.Tensor = aggr(in_attrs).val
    out_val = out_tensor.detach().numpy()

    return np.allclose(out_val, out_target)


class TestSumAggregation(TestCase):
    @staticmethod
    def _feed(in_val: List[List[float]], out_target: List[float]) -> bool:
        return _feed(SumAggregation(), in_val, out_target)

    def test_zero_attributes(self):
        with self.assertRaises(AssertionError):
            self._feed([], [])

    def test_single_attribute_identity(self):
        self.assertTrue(self._feed(in_val=[[1., 2., 3.]], out_target=[1., 2., 3.]))
        self.assertTrue(self._feed(in_val=[[1.]], out_target=[1.]))
        self.assertTrue(self._feed(in_val=[[-501., 0., 0.]], out_target=[-501., 0., 0.]))

    def test_multiple_attributes(self):
        self.assertTrue(self._feed(in_val=[[1., 2., 3.], [1., 2., 3.]],
                                   out_target=[2., 4., 6.]))

        self.assertTrue(self._feed(in_val=[[0., 0., 0.], [1., 1., 1.], [-4., -5.1, -100.234]],
                                   out_target=[-3., -4.1, -99.234]))


class TestAvgAggregation(TestCase):
    @staticmethod
    def _feed(in_val: List[List[float]], out_target: List[float]) -> bool:
        return _feed(AvgAggregation(), in_val, out_target)

    def test_zero_attributes(self):
        with self.assertRaises(AssertionError):
            self._feed([], [])

    def test_single_attribute_identity(self):
        self.assertTrue(self._feed(in_val=[[1., 2., 3.]], out_target=[1., 2., 3.]))
        self.assertTrue(self._feed(in_val=[[1.]], out_target=[1.]))
        self.assertTrue(self._feed(in_val=[[-501., 0., 0.]], out_target=[-501., 0., 0.]))

    def test_multiple_attributes(self):
        self.assertTrue(self._feed(in_val=[[1., 2., 3.], [1., 2., 3.]],
                                   out_target=[1., 2., 3.]))

        self.assertTrue(self._feed(in_val=[[0., 0., 0.], [1., 1., 1.], [-4., -5.1, -100.234]],
                                   out_target=[-3/3., -4.1/3, -99.234/3]))

from unittest import TestCase

from torch import Tensor

from graph_nets.utils import tensors_stackable


class TestStackable(TestCase):
    def test_valid_case(self):
        self.assertTrue(tensors_stackable(tensors=[Tensor([[1, 2, 3], [1, 2, 3]]), Tensor([[3, 4, 5], [4, 5, 6]])]))
        self.assertTrue(tensors_stackable(tensors=[Tensor([1, 2, 3]), Tensor([3, 4, 5])]))
        self.assertTrue(tensors_stackable(tensors=[Tensor([1, 2, 3])]))

    def test_invalid_case(self):
        self.assertFalse(tensors_stackable(tensors=[Tensor([[1, 2, 3], [1, 2, 3]]), Tensor([[8, 9]])]))
        self.assertFalse(tensors_stackable(tensors=[Tensor([1, 2]), Tensor([1])]))

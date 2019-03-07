from typing import List, Callable

import torch
import numpy as np
from unittest import TestCase

from rank_predictor.model.util import global_avg_pool_1d, global_avg_pool, flatten


class TestUtil(TestCase):
    @staticmethod
    def feed_and_expect(method: Callable[[torch.Tensor], torch.Tensor],
                        input_tensor: List[List[List]], target_tensor: List[List]) -> bool:
        """
        Feed into `method` and compare the output numerically to the target.
        :return: True if feeding `input_tensor` into `method` yields something close to `target_tensor`.
        """
        input_tensor = np.array(input_tensor, dtype=np.float32)
        target_tensor = np.array(target_tensor, dtype=np.float32)
        tensor_3d = torch.Tensor(input_tensor).float()
        tensor_pooled = method(tensor_3d)
        tensor_pooled = tensor_pooled.detach().numpy()
        return np.allclose(target_tensor, tensor_pooled)

    def test_1d_global_avg_pool_invalid_input(self):
        with self.assertRaises(AssertionError):
            global_avg_pool_1d(torch.Tensor([[1, 2], [3, 4]]))

    def test_1d_global_avg_pool(self):
        self.assertTrue(self.feed_and_expect(global_avg_pool_1d,
                                             input_tensor=[[[1], [4.1]]],
                                             target_tensor=[[1, 4.1]]))
        self.assertTrue(self.feed_and_expect(global_avg_pool_1d,
                                             input_tensor=[[[1, 2], [1, 4]]],
                                             target_tensor=[[1.5, 2.5]]))
        self.assertTrue(self.feed_and_expect(global_avg_pool_1d,
                                             input_tensor=[[[1, 2], [1, 4]], [[10, 20], [10, 40]]],
                                             target_tensor=[[1.5, 2.5], [15., 25.]]))

    def test_global_avg_pool(self):
        self.assertTrue(self.feed_and_expect(global_avg_pool,
                                             input_tensor=[[[[1, 2], [3, 4]]]],
                                             target_tensor=[[(1 + 2 + 3 + 4) / 4]]))
        self.assertTrue(self.feed_and_expect(global_avg_pool,
                                             input_tensor=[[[[1, 2], [3, 4]], [[4, 5], [6, 7]]]],
                                             target_tensor=[[(1 + 2 + 3 + 4) / 4, (4 + 5 + 6 + 7) / 4]]))
        self.assertTrue(self.feed_and_expect(global_avg_pool,
                                             input_tensor=[[[[[[[[1, 3]]]]]]]],
                                             target_tensor=[[2]]))

    def test_flatten(self):
        self.assertTrue(self.feed_and_expect(flatten,
                                             input_tensor=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                             target_tensor=[[1, 2, 3, 4], [5, 6, 7, 8]]))

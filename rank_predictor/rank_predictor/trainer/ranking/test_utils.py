from typing import List, Union
import numpy as np
import torch
from unittest import TestCase
from rank_predictor.trainer.ranking.utils import compute_batch_accuracy, compute_multi_batch_accuracy


class TestUtils(TestCase):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def _feed_multi_batch_accuracy(self, target_ranks_list: List[List[Union[float, int]]],
                                   model_outputs_list: List[List[float]], accuracy_target: float,
                                   correct_ctr_target: int) -> None:
        target_ranks_list = list(map(torch.Tensor, target_ranks_list))
        model_outputs_list = list(map(torch.Tensor, model_outputs_list))

        accuracy, correct_ctr = compute_multi_batch_accuracy(target_ranks_list, model_outputs_list)
        accuracy = accuracy.numpy()
        correct_ctr = correct_ctr.numpy()

        self.assertTrue(np.isclose(accuracy, accuracy_target))
        self.assertTrue(correct_ctr == correct_ctr_target)

    def _feed_batch_accuracy(self, target_ranks: List[Union[float, int]], model_outputs: List[float],
                             accuracy_target: float, correct_ctr_target: int) -> None:
        target_ranks = torch.Tensor(target_ranks)
        model_outputs = torch.Tensor(model_outputs)

        accuracy, correct_ctr = compute_batch_accuracy(target_ranks, model_outputs)
        accuracy = accuracy.numpy()
        correct_ctr = correct_ctr.numpy()

        self.assertTrue(np.isclose(accuracy, accuracy_target))
        self.assertTrue(correct_ctr == correct_ctr_target)

    def test_batch_accuracy_all_correct(self) -> None:
        n = 5
        self._feed_batch_accuracy(target_ranks=[1, 2, 3, 4, 5],
                                  model_outputs=[.5, .4, .3, .2, .1],
                                  accuracy_target=1.0,
                                  correct_ctr_target=n * n - n)

    def test_batch_accuracy_all_wrong(self) -> None:
        self._feed_batch_accuracy(target_ranks=[1, 2, 3, 4, 5],
                                  model_outputs=[0.1, 0.2, 0.3, 0.4, 0.5],
                                  accuracy_target=.0,
                                  correct_ctr_target=0)

    def test_batch_accuracy_some_correct(self) -> None:
        self._feed_batch_accuracy(target_ranks=[3, 2, 1, 4, 5],
                                  model_outputs=[.2, .3, .4, .1, 1.],
                                  accuracy_target=12 / 20,
                                  correct_ctr_target=20 - 4 - 4)

    def test_multi_batch_accuracy_all_correct(self):
        n = 5
        self._feed_multi_batch_accuracy(target_ranks_list=[[1, 2], [3, 4], [5]],
                                        model_outputs_list=[[.5, .4], [.3, .2], [.1]],
                                        accuracy_target=1.0,
                                        correct_ctr_target=n * n - n)

    def test_multi_batch_accuracy_some_correct(self):
        self._feed_multi_batch_accuracy(target_ranks_list=[[3, 2], [1, 4], [5]],
                                        model_outputs_list=[[.2, .3], [.4, .1], [1.]],
                                        accuracy_target=12 / 20,
                                        correct_ctr_target=20 - 4 - 4)

from unittest import TestCase
import numpy as np
import torch

from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss


class TestProbabilisticLoss(TestCase):
    def test_ground_truth_matrix(self):
        # arrange
        input = torch.Tensor([1, 2, 3, 4, 4.5, 0])
        target_out = np.array(
            [[0.5, 1, 1, 1, 1, 0],
             [0, 0.5, 1, 1, 1, 0],
             [0, 0, 0.5, 1, 1, 0],
             [0, 0, 0, 0.5, 1, 0],
             [0, 0, 0, 0, 0.5, 0],
             [1, 1, 1, 1, 1, 0.5]])

        # act
        out = ProbabilisticLoss.ground_truth_matrix(input)
        out = out.numpy()

        # assert
        match = np.array_equal(out, target_out)
        self.assertTrue(match)

    def test_model_prediction_matrix_discretization(self):
        # arrange
        f = torch.Tensor([-.5, .1, .5, -1.])
        target_out = np.array(
            [[0.5, 0, 0, 1],
             [1, 0.5, 0, 1],
             [1, 1, 0.5, 1],
             [0, 0, 0, 0.5]])

        # act
        out = ProbabilisticLoss.discretize_model_prediction_matrix(ProbabilisticLoss.model_prediction_matrix(f))
        out = out.numpy()

        # assert
        match = np.array_equal(out, target_out)
        self.assertTrue(match)

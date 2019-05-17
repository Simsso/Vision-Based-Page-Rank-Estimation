from typing import List
import torch
from torch import Tensor

from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss


def compute_multi_batch_accuracy(target_ranks_batches: List[Tensor], model_outputs_batches: List[Tensor])\
        -> (Tensor, Tensor):
    """
    Computes the ration of correct relative rank predictions for a list of batches. Opposed to calling
    `compute_batch_accuracy` for every single batch, the function compares pair-wise for all combinations.
    That is the same as `compute_batch_accuracy(flatten(target_ranks), flatten(model_output_batches))`.
    :param target_ranks_batches: List of target rank batches, length len
    :param model_outputs_batches: List of output batches, length len, greater values indicate lower ranks,
                                 e.g. f(x_1) > f(x_2) indicates that x_1 has a higher rank, say #2, and x_2 #8.
    :return: Ratio of correct predictions (aka. accuracy) and number of correct predictions (pairwise).
             Note that the num_correct is in {0, ..., n*n - n}
    """
    n = len(target_ranks_batches)
    assert n == len(model_outputs_batches), "Number of rank and output batches must be identical"
    assert n > 0, "Cannot compute the accuracy for zero samples"

    return compute_batch_accuracy(torch.cat(target_ranks_batches, dim=0), torch.cat(model_outputs_batches, dim=0))


def compute_batch_accuracy(target_ranks: Tensor, model_outputs: Tensor) -> (Tensor, Tensor):
    """
    Computes the ratio of correct relative rank predictions within a batch.
    :param target_ranks: Target ranks of the samples in a batch of size n
    :param model_outputs: Model predictions for the batch, greater values indicate lower ranks,
                          e.g. f(x_1) > f(x_2) indicates that x_1 has a higher rank, say #2, and x_2 #8.
    :return: Ratio of correct predictions (aka. accuracy) and number of correct predictions (pairwise).
             Note that the num_correct is in {0, ..., n*n - n}
    """
    assert len(target_ranks.size()) == 1, "Target ranks must be a vector, i.e. shape [n]"
    assert len(model_outputs.size()) == 1, "Model outputs must be a vector, i.e. shape [n]"

    n = target_ranks.size(0)

    assert n == model_outputs.size(0), "Input shapes must match"
    assert n > 1, "Batch size must be greater than 1 for a ranking comparison to be computed"

    # transpose for broadcasting
    target_ranks_t = target_ranks.view((1, n))
    model_outputs_t = model_outputs.view((1, n))
    target_ranks = target_ranks.view((n, 1))
    model_outputs = model_outputs.view((n, 1))

    # check where targets are less than or equal (and where the model output is)
    target_lt = torch.lt(target_ranks, target_ranks_t)
    model_gt = torch.gt(model_outputs, model_outputs_t)

    correct_pairs = torch.eq(target_lt, model_gt)
    num_correct = torch.sum(correct_pairs) - n  # ignore diagonal entries (x_i ranked vs. x_i)

    correct_ratio = num_correct.float() / (n ** 2 - n)

    return correct_ratio, num_correct


def per_sample_accuracy(r: Tensor, f: Tensor) -> Tensor:
    """
    Computes the accuracy the model hat for a given sample wrt. to all other samples (including itself).
    :param r: Relative ranking of the samples, where r_i < r_j indicates that x_i is supposed to be ranked higher
              than x_j, e.g. x_i is at position 10 and x_j at position 15. The scaling can be linear, log, ...
              Vector size [n]
    :param f: For every sample x_i in the batch, with i in {1, ..., n}, f_i is the network output f(x_i). Size [n].
    :return: n-dim vector which accuracy scores for the i-th sample in [0,1].
    """

    n = r.size(0)
    assert r.shape == f.shape, "Shapes of 'r' and 'f' must be equal."

    # ground truth and prediction matrices
    gt = ProbabilisticLoss.ground_truth_matrix(r)
    pred = ProbabilisticLoss.discretize_model_prediction_matrix(ProbabilisticLoss.model_prediction_matrix(f))

    # number of matches
    correct = torch.eq(gt, pred).float()
    correct_ctr = torch.sum(correct, dim=1, keepdim=False)

    # accuracy
    acc = correct_ctr / n

    return acc

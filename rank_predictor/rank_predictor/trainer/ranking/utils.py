import torch
from torch import Tensor


def compute_accuracy(target_ranks: Tensor, model_outputs: Tensor) -> (Tensor, Tensor):
    """
    Computes the ratio of correct relative rank predictions within a batch.
    :param target_ranks: Target ranks of the samples in a batch
    :param model_outputs: Model predictions for the batch
    :return: Ratio of correct predictions (aka. accuracy) and number of correct predictions (pairwise)
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
    model_lt = torch.lt(model_outputs, model_outputs_t)

    correct_pairs = torch.eq(target_lt, model_lt)
    num_correct = torch.sum(correct_pairs) - n

    correct_ratio = num_correct.float() / (n-1) ** 2

    return correct_ratio, num_correct

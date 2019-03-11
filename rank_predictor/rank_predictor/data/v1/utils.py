import torch
from torch import Tensor
from skimage import io


def img_loading_possible(img_path: str) -> bool:
    try:
        io.imread(img_path)
        return True
    except ValueError:
        return False


def compute_accuracy(target_ranks: Tensor, model_outputs: Tensor) -> (Tensor, Tensor):
    """
    Computes the ratio of correct relative rank predictions within a batch.
    :param target_ranks: Target ranks of the samples in a batch
    :param model_outputs: Model predictions for the batch
    :return: Ratio of correct predictions (aka. accuracy) and number of correct predictions (pairwise)
    """
    assert len(target_ranks.size()) == 1, "Target ranks must be a vector, i.e. shape [n]"
    assert len(model_outputs.size()) == 1, "Model outputs must be a vector, i.e. shape [n]"

    num_samples = target_ranks.size(0)

    assert num_samples == model_outputs.size(0), "Input shapes must match"

    # transpose for broadcasting
    target_ranks_t = target_ranks.view((1, num_samples))
    model_outputs_t = model_outputs.view((1, num_samples))
    target_ranks = target_ranks.view((num_samples, 1))
    model_outputs = model_outputs.view((num_samples, 1))

    # check where targets are less than or equal (and where the model output is)
    target_lt = torch.lt(target_ranks, target_ranks_t)
    model_lt = torch.lt(model_outputs, model_outputs_t)

    correct_pairs = torch.eq(target_lt, model_lt)
    num_correct = torch.sum(correct_pairs)

    correct_ratio = num_correct.float() / num_samples ** 2

    return correct_ratio, num_correct

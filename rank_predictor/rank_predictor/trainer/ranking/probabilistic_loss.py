import torch
from torch import Tensor


class ProbabilisticLoss:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, f: Tensor, r: Tensor, w: Tensor) -> Tensor:
        """
        Loss for a batch of samples.
        :param f: For every sample x_i in the batch, with i in {1, ..., n}, f_i is the network output f(x_i).
        :param r: Relative ranking of the samples, where r_i < r_j indicates that x_i is supposed to be ranked higher
                  than x_j, e.g. x_i is at position 10 and x_j at position 15. The scaling can be linear, log, ...
        :param w: Weights of the samples
        :return: Float tensor representing the loss
        """

        assert len(f.size()) == 1, "Model output must be a vector, i.e. shape [n]"
        assert len(r.size()) == 1, "Target ranks must be a vector, i.e. shape [n]"

        n = f.size(0)

        assert n == r.size(0), "Input shapes must match"

        # transpose for broadcasting
        r_t = r.view((1, n))
        f_t = f.view((1, n))
        w_t = w.view((1, n))
        r = r.view((n, 1))
        f = f.view((n, 1))
        w = w.view((n, 1))

        # compute ground truth matrix
        # p_ij = 1   <==> x_i has better rank than x_j
        # p_ij = 0   <==> x_i has lower rank than x_j
        # p_ij = 0.5 <==> x_i has equal rank as x_j, i.e. i = j
        p_ij = torch.lt(r, r_t).float()
        p_ii_vals = 0.5 * torch.eye(n, device=p_ij.device)
        p_ij = p_ij + p_ii_vals

        # compute model predictions
        # o_ij = f(x_i) - f(x_j)
        o_ij = f - f_t

        # compute cost for each prediction
        c_ij = -p_ij * o_ij + torch.log(1 + torch.exp(o_ij))

        # weighting of each prediction is w_i * w_j
        # w_ij = w * w_t
        c_ij = c_ij * w   # apply weighting

        # compute total cost (normalize by n**2)
        c = torch.sum(c_ij) / (n**2)

        return c

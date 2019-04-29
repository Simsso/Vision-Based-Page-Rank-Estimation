import torch
from torch import Tensor


class ProbabilisticLoss:

    def __init__(self, scaling_fac: float = 1.0) -> None:
        """
        :param scaling_fac: Factor by which the loss will be scaled.
        """
        super().__init__()
        self.scaling_fac = scaling_fac

    @staticmethod
    def ground_truth_matrix(r: Tensor) -> Tensor:
        """
        Computes the ground truth matrix given a rank vector.
        :param r: Relative ranking of the samples, where r_i < r_j indicates that x_i is supposed to be ranked higher
                  than x_j, e.g. x_i is at position 10 and x_j at position 15. The scaling can be linear, log, ...
                  Vector size [n]
        :return: Matrix of size [n,n], where entries are in {0, .5, 1} (lower, same, higher rank). Interpretation:
                 out_ij = 0   <=> sample i has a lower rank (e.g. #50) than sample j (e.g. #2)
                 out_ij = 0.5 <=> sample i has the same rank (e.g. #5) as sample j (e.g. #5)
                 out_ij = 1   <=> sample i has a better rank (e.g. #5) than sample j (e.g. #100)
        """

        n = r.size(0)

        # transpose for broadcasting
        r_t = r.view((1, n))
        r = r.view((n, 1))

        # compute ground truth matrix
        # p_ij = 1   <==> x_i has better rank than x_j
        # p_ij = 0   <==> x_i has lower rank than x_j
        # p_ij = 0.5 <==> x_i has equal rank as x_j, i.e. i = j
        p_ij = torch.lt(r, r_t).float()
        p_ii_vals = 0.5 * torch.eye(n, device=p_ij.device)
        p_ij = p_ij + p_ii_vals

        return p_ij

    @staticmethod
    def model_prediction_matrix(f: Tensor) -> Tensor:
        """
        Computes the model prediction matrix given a model outputs vector.
        :param f: For every sample x_i in the batch, with i in {1, ..., n}, f_i is the network output f(x_i). Size [n].
        :return: Matrix of size [n,n], entries correspond to the way the model ranks the samples:
                 < 0 lower, == 0 same, > 0 higher.
        """

        n = f.size(0)

        # transpose for broadcasting
        f_t = f.view((1, n))
        f = f.view((n, 1))

        # compute model predictions
        # o_ij = f(x_i) - f(x_j)
        o_ij = f - f_t

        return o_ij

    @staticmethod
    def discretize_model_prediction_matrix(f_mat: Tensor) -> Tensor:
        """
        Discretizes a model prediction matrix in this manner:
        out_ij := 0   if f_mat_ij < 0
        out_ij := 0.5 if f_mat_ij == 0.5
        out_ij := 1   otherwise
        :param f_mat: Input matrix computed by `model_prediction_matrix`, size [n,n].
        :return: Discrete version of f_mat.
        """

        lt = torch.lt(f_mat, 0.).float()
        eq = torch.eq(f_mat, 0.).float()
        gt = torch.gt(f_mat, 0.).float()

        out = lt * 0 + eq * 0.5 + gt * 1

        return out

    def __call__(self, f: Tensor, r: Tensor, w: Tensor) -> Tensor:
        """
        Loss for a batch of samples.
        :param f: For every sample x_i in the batch, with i in {1, ..., n}, f_i is the network output f(x_i).
        :param r: Relative ranking of the samples, where r_i < r_j indicates that x_i is supposed to be ranked higher
                  than x_j, e.g. x_i is at position 10 and x_j at position 15. The scaling can be linear, log, ...
        :param w: Weights of the samples
        :return: Float scalar tensor representing the loss
        """

        assert len(f.size()) == 1, "Model output must be a vector, i.e. shape [n]"
        assert len(r.size()) == 1, "Target ranks must be a vector, i.e. shape [n]"

        n = f.size(0)

        assert n == r.size(0), "Input shapes must match"

        # transpose for broadcasting
        w_t = w.view((1, n))
        w = w.view((n, 1))

        p_ij = ProbabilisticLoss.ground_truth_matrix(r)

        o_ij = ProbabilisticLoss.model_prediction_matrix(f)

        # compute cost for each prediction
        c_ij = -p_ij * o_ij + torch.log(1 + torch.exp(o_ij))

        # weighting of each prediction is w_i * w_j
        # w_ij = w * w_t
        c_ij = c_ij   # apply weighting

        # compute total cost (normalize by n**2)
        c = torch.sum(c_ij) / (n**2)

        c *= self.scaling_fac

        return c

import torch
from torch import Tensor


class SquaredErrorLoss:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return torch.sum(weight * (prediction - target) ** 2)

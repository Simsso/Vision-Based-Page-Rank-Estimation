import torch
import torch.nn.functional as F
from torch import Tensor


def global_avg_pool_1d(x: Tensor) -> Tensor:
    """
    Global 1-d average pooling, shortcut method for 1d average pooling with kernel size = width (w)
    :param x: Input tensor with shape [b, c, w]
    :return: Pooled tensor of shape [b, c]
    """
    assert len(x.shape) == 3, "Input tensor must have shape [b, c, w], got {}".format(x.shape)

    c, w = x.shape[1:]
    x = F.avg_pool1d(x, kernel_size=w)

    x = x.view((-1, c))  # remove width dimension

    return x


def global_avg_pool(x: Tensor) -> Tensor:
    """
    Global n-d average pooling.
    :param x: Input tensor with shape [b, c, d1, ..., dn], where n >= 1
    :return:Pooled tensor of shape [b, c]
    """
    assert len(x.shape) >= 3, "Input tensor must have shape [b, c, d1, ..., dn], where n >= 1, got {}".format(x.shape)

    b = x.size(0)
    c = x.size(1)

    x = x.view((b, c, -1))

    return torch.mean(x, dim=2)


def flatten(x: Tensor) -> Tensor:
    """
    Flattens a given tensor
    :param x: Input tensor of shape [b, d1, ..., dn], where n >= 1
    :return: Output tensor of shape [b, d1*...*dn]
    """
    b = x.size(0)
    x = x.view(b, -1)
    return x

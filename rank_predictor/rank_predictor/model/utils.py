import operator
from functools import reduce
from typing import Dict

import torch
import torch.nn.functional as F
from graph_nets.functions.update import IndependentNodeUpdate

from graph_nets.block import GNBlock
from torch import Tensor, nn


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


def get_extraction_block(model: torch.nn.Module):
    """
    Creates a new extraction block which operates on a raw graph as provided by the dataset.
    Applies the model to each node.
    """
    def node_extractor_fn(node: Dict[str, any]) -> Tensor:
        desktop_img: Tensor = node['desktop_img']
        mobile_img: Tensor = node['mobile_img']

        # desktop and mobile feature vector
        x1, x2 = model(
            desktop_img.unsqueeze(0),
            mobile_img.unsqueeze(0)
        )

        x = torch.cat((x1, x2), dim=1).view(-1)

        return x

    return GNBlock(
        phi_v=IndependentNodeUpdate(node_extractor_fn))


class ListModule(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    Can be used to have nn.Modules which contain lists of nn.Modules.
    """

    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def forward(self, *args):
        raise NotImplementedError()


def parameter_count(model: nn.Module) -> int:
    """
    :param model: A PyTorch module.
    :return: Number of parameters in the passed module.
    """
    params = [p for p in model.parameters()]
    param_count = map(lambda p: p.size(), params)
    param_count = map(lambda p_size: reduce(operator.mul, p_size, 1), param_count)
    param_count = sum(param_count)
    return param_count

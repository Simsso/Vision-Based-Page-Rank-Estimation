import torch
import torch.nn.functional as F


def global_avg_pool_1d(x: torch.Tensor) -> torch.Tensor:
    feature_size = x.size(1)
    return F.avg_pool1d(x, kernel_size=feature_size)


def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    assert len(x.size()) >= 3

    batch_size = x.size(0)
    num_channels = x.size(1)

    x = x.view((batch_size, num_channels, -1))

    return torch.mean(x, dim=2)


def flatten(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size(0)
    return x.view(batch_size, -1)

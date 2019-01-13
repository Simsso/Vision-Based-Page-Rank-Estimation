import torch
import torch.nn.functional as F


def global_avg_pool_1d(x: torch.Tensor) -> torch.Tensor:
    feature_size = x.size(1)
    return F.avg_pool1d(x, kernel_size=feature_size)


def global_avg_pool(x: torch.Tensor) -> torch.Tensor:
    assert len(x.size()) in [2, 3]

    batch_size = x.size(0)
    if len(x.size()) == 2:
        x = x.view((batch_size, 1, -1))
    return F.adaptive_avg_pool1d(x, output_size=(batch_size,))


def flatten(x: torch.Tensor) -> torch.Tensor:
    batch_size = x.size(0)
    return x.view(batch_size, -1)

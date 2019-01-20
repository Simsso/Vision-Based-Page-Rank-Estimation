import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class TrainingRun:
    def __init__(self,
                 net: nn.Module,
                 opt: optim.Optimizer,
                 loss_fn,
                 data: DataLoader) -> None:
        self.net = net
        self.opt = opt
        self.loss_fn = loss_fn
        self.data = data
        self.step_ctr = 0

        self.loss_log = []

    def __call__(self, epochs: int) -> None:
        for epoch in range(epochs):
            for batch in self.data:
                imgs = batch['img']
                ranks = batch['rank']
                self._step(imgs, ranks)

    def _step(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.step_ctr += 1
        self.opt.zero_grad()
        model_out = self.net.forward(inputs)
        loss = self.loss_fn(model_out, targets.float() / 1e5)
        loss.backward()
        self._log_loss(loss)
        self.opt.step()

    def _log_loss(self, loss) -> None:
        self.loss_log.append(loss.item())
        print(self.loss_log)

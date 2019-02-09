import logging
import multiprocessing

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from rank_predictor.data import threefold


class TrainingRun:
    def __init__(self,
                 net: nn.Module,
                 opt: optim.Adam,
                 loss_fn,
                 data: threefold.Data,
                 batch_size: int,
                 device) -> None:
        self.net = net
        self.opt = opt
        self.loss_fn = loss_fn
        self.step_ctr = 0
        self.device = device

        cpu_count = multiprocessing.cpu_count()
        worker_count = max(cpu_count - 1, 1)
        logging.info("Using {} workers for the data pipeline".format(worker_count))

        self.data = data

        # create data loader from dataset
        self.data_loader: threefold.Data[DataLoader] = threefold.Data(
            train=DataLoader(data.train, batch_size, shuffle=True, num_workers=worker_count),
            valid=DataLoader(data.valid, batch_size, shuffle=False, num_workers=worker_count),
            test=DataLoader(data.test, batch_size, shuffle=False, num_workers=worker_count),
        )

        self.loss_log = []

        self.net.to(device)

        self.writer = SummaryWriter('logs')

    def __call__(self, epochs: int) -> None:
        for epoch in range(epochs):
            logging.info("Starting epoch #{}".format(epoch + 1))
            for batch in self.data_loader.train:
                if self.step_ctr % 128 == 0:
                    self._run_valid()

                imgs = batch['img'].to(self.device)
                logranks = batch['logrank'].to(self.device).float()
                self._train_step(imgs, logranks)

    def _train_step(self, inputs: torch.Tensor, logranks: torch.Tensor) -> None:
        self.net.train()

        self.step_ctr += 1
        self.opt.zero_grad()

        model_out: torch.Tensor = self.net.forward(inputs)

        loss = self.loss_fn(model_out, logranks)
        loss.backward()
        self.opt.step()

        self.writer.add_scalar('loss_train', loss, self.step_ctr)

    def _run_valid(self) -> None:
        logging.info("Running validation")

        self.net.eval()

        # accumulators
        loss_sum = 0.

        for batch in tqdm(self.data_loader.valid):
            imgs: torch.Tensor = batch['img'].to(self.device)
            logranks: torch.Tensor = batch['logrank'].to(self.device).float()

            # forward pass
            with torch.no_grad():
                model_out: torch.Tensor = self.net.forward(imgs)

            loss = self.loss_fn(model_out, logranks)
            loss_sum += loss

        n = len(self.data_loader.valid)
        loss = loss_sum / n

        self.writer.add_scalar('loss_valid', loss, self.step_ctr)

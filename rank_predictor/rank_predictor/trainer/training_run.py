import logging
import multiprocessing
from typing import Dict, Callable, Union, List
import sacred
from tqdm import tqdm
from graph_nets import Graph
import torch
import numpy as np
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from rank_predictor.trainer.ranking.utils import compute_batch_accuracy, compute_multi_batch_accuracy
from rank_predictor.data import threefold


class TrainingRun:
    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn: Callable,
                 data: threefold.Data, batch_size: int, device: torch.device, collate_fn: Callable = None) -> None:
        self.ex = ex
        self.net = net
        self.opt = opt
        self.loss_fn = loss_fn
        self.step_ctr = 0
        self.device = device
        self.batch_size = batch_size

        cpu_count = multiprocessing.cpu_count()
        worker_count = 1  # max(cpu_count - 1, 1)
        logging.info("Using {} workers for the data pipeline".format(worker_count))

        self.data = data

        # create data loader from dataset
        self.data_loader: threefold.Data[DataLoader] = threefold.Data(
            train=DataLoader(data.train, batch_size, shuffle=True, num_workers=worker_count, collate_fn=collate_fn),
            valid=DataLoader(data.valid, 1, shuffle=True, num_workers=worker_count, collate_fn=collate_fn),
            test=DataLoader(data.test, batch_size, shuffle=False, num_workers=worker_count, collate_fn=collate_fn),
        )

        self.loss_log = []

        self.net.to(device)

        self.writer = SummaryWriter('logs/{}'.format(name))

    def __call__(self, epochs: int) -> None:
        for epoch in range(epochs):
            logging.info("Starting epoch #{}".format(epoch + 1))
            for batch in tqdm(self.data_loader.train):
                if self.step_ctr % 500 == 0:
                    self._run_valid(self.data_loader.valid, 'valid', approx=True)

                self.step_ctr += 1
                self._train_step(batch)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> None:
        raise NotImplementedError

    def log_scalar(self, name: str, val: Union[np.ndarray, torch.Tensor]):
        # tensorboard logging
        self.writer.add_scalar(name, val, self.step_ctr)

        # sacred logging
        if isinstance(val, torch.Tensor):
            val = float(val.detach().numpy())
        self.ex.log_scalar(name, val, self.step_ctr)


class GNTrainingRun(TrainingRun):

    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn, data: threefold.Data, batch_size: int, device) -> None:
        def collate_fn(batch):
            return batch

        super().__init__(ex, name, net, opt, loss_fn, data, batch_size, device, collate_fn)

    def _train_step(self, batch: List[Dict[str, Union[int, Graph]]]) -> None:
        self.net.train()
        self.opt.zero_grad()

        model_outs, logranks = [], []

        for sample in batch:
            logrank: float = sample['logrank']
            graph: Graph = sample['graph']
            graph.to(self.device)

            try:
                model_out: torch.Tensor = self.net.forward(graph)
            except RuntimeError:
                logging.warning("Skipping step because of runtime error")
                self.step_ctr -= 1

                # free GPU memory
                self.opt.zero_grad()
                self.net.cpu()
                torch.cuda.empty_cache()
                self.net.cuda(self.device)

                return

            model_outs.append(model_out)
            logranks.append(logrank)

        model_outs = torch.cat(model_outs).to(self.device)
        logranks = torch.Tensor(logranks).to(self.device).float()

        loss = self.loss_fn(model_outs, logranks, w=(1-logranks))
        loss.backward()  # TODO: double-check that gradients of all samples will be taken into account

        self.opt.step()

        accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_outs)

        self.log_scalar('batch_loss_train', loss)
        self.log_scalar('batch_accuracy_train', accuracy)
        self.writer.add_histogram('batch_model_out_train', model_outs, self.step_ctr)
        self.writer.add_histogram('batch_model_target_train', logranks, self.step_ctr)

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> None:
        self.opt.zero_grad()
        self.net.eval()

        with torch.no_grad():
            # accumulators
            model_outs, logranks = [], []

            for batch in dataset:
                if approx and len(model_outs) >= 10:
                    break
                for sample in batch:
                    logrank: float = sample['logrank']
                    graph: Graph = sample['graph']
                    graph.to(self.device)
                    model_out: torch.Tensor = self.net.forward(graph)

                    model_outs.append(model_out)
                    logranks.append(logrank)

            model_outs = torch.cat(model_outs).to(device='cpu')
            logranks = torch.Tensor(logranks).to(device='cpu')

            loss = self.loss_fn(model_outs, logranks, w=(1-logranks))

            accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_outs)

            self.log_scalar('loss_{}'.format(name), loss)
            self.log_scalar('accuracy_{}'.format(name), accuracy,)


class VanillaTrainingRun(TrainingRun):

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        self.net.train()
        self.opt.zero_grad()

        inputs = batch['img'].to(self.device)
        logranks = batch['logrank'].to(self.device).float()

        model_out: torch.Tensor = self.net.forward(inputs)

        loss = self.loss_fn(model_out, logranks, w=(1-logranks))
        loss.backward()
        self.opt.step()

        accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_out)

        self.log_scalar('batch_loss_train', loss)
        self.log_scalar('batch_accuracy_train', accuracy)
        self.writer.add_histogram('batch_model_out_train', model_out, self.step_ctr)
        self.writer.add_histogram('batch_model_target_train', logranks, self.step_ctr)

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> None:
        self.net.eval()

        with torch.no_grad():
            # accumulators
            loss_sum, model_out_batches, rank_batches = 0., [], []

            for batch in dataset:
                imgs: torch.Tensor = batch['img'].to(self.device)
                logranks: torch.Tensor = batch['logrank'].to(self.device).float()
                rank_batches.append(logranks)

                # forward pass
                with torch.no_grad():
                    model_out: torch.Tensor = self.net.forward(imgs)
                    model_out_batches.append(model_out)

                    loss = self.loss_fn(model_out, logranks, w=(1-logranks))
                    loss_sum += loss

            n = len(dataset)
            loss = loss_sum / n

            accuracy, _ = compute_multi_batch_accuracy(rank_batches, model_out_batches)

            self.log_scalar('loss_{}'.format(name), loss)
            self.log_scalar('accuracy_{}'.format(name), accuracy)

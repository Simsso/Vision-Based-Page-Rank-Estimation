import logging
import multiprocessing
import os
from typing import Dict, Callable, Union, List
import sacred
import torch
import numpy as np
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from graph_nets.data_structures.graph import Graph
from rank_predictor.trainer.ranking.utils import compute_batch_accuracy, compute_multi_batch_accuracy
from rank_predictor.data import threefold


class TrainingRun:
    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn: Callable,
                 data: threefold.Data, batch_size: int, device: torch.device, collate_fn: Callable = None) -> None:
        self.ex = ex
        self.name = name
        self.net = net
        self.opt = opt
        self.loss_fn = loss_fn
        self.step_ctr = 0
        self.device = device
        self.batch_size = batch_size

        save_dir = os.path.expanduser(os.environ['model_save_dir'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

        cpu_count = multiprocessing.cpu_count()
        worker_count = 0  # max(cpu_count - 1, 1)
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

    def __call__(self, epochs: int) -> float:
        """
        :param epochs: Number of epochs to train for.
        :return: Final validation accuracy
        """
        for epoch in range(epochs):
            self._save_model(epoch)
            logging.info("Starting epoch #{}".format(epoch + 1))
            for batch in self.data_loader.train:
                if self.step_ctr % 5000 == 0:
                    logging.info("Running approx. validation at step #{}".format(self.step_ctr))
                    self._run_valid(self.data_loader.valid, 'valid', approx=True)
                    self._run_valid(self.data_loader.train, 'train', approx=True)

                self.step_ctr += 1
                self._train_step(batch)
        self._save_model(epochs)
        return self._run_valid(self.data_loader.valid, 'valid', approx=False)

    def _save_model(self, epoch: int) -> None:
        save_dir = self.save_dir
        file_name = '{}_{:04d}.pt'.format(self.name, epoch)
        save_path = os.path.join(save_dir, file_name)

        logging.info("Storing model at '{}'.".format(save_path))

        torch.save(self.net.state_dict(), save_path)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
        raise NotImplementedError

    def log_scalar(self, name: str, val: Union[np.ndarray, torch.Tensor]):
        # tensorboard logging
        self.writer.add_scalar(name, val, self.step_ctr)

        # sacred logging
        if isinstance(val, torch.Tensor):
            val = float(val.cpu().detach().numpy())
        self.ex.log_scalar(name, val, self.step_ctr)


class GNTrainingRun(TrainingRun):

    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn, data: threefold.Data,
                 batch_size: int, pairwise_batch_size: int, device) -> None:
        def collate_fn(batch):
            return batch

        super().__init__(ex, name, net, opt, loss_fn, data, batch_size, device, collate_fn)

        self.pairwise_batch_size = pairwise_batch_size

    def _train_step(self, batch: List[Dict[str, Union[int, Graph]]]) -> None:
        self.net.train()
        self.opt.zero_grad()

        model_outs, logranks = [], []

        for sample in batch:
            logrank: float = sample['logrank']
            graph: Graph = sample['graph']
            graph.to(self.device)

            model_out: torch.Tensor = self.net.forward(graph)

            model_outs.append(model_out)
            logranks.append(logrank)

        # increase the pairwise batch with samples w/o gradient (to save RAM but increase the gradient precision)
        # TODO: disable dropout for those samples
        with torch.no_grad():
            for pairwise_batch in self.data_loader.train:
                for sample in pairwise_batch:
                    if len(model_outs) >= self.pairwise_batch_size:
                        break
                    logrank: float = sample['logrank']
                    graph: Graph = sample['graph']
                    graph.to(self.device)
                    model_out: torch.Tensor = self.net.forward(graph)

                    logranks.append(logrank)
                    model_outs.append(model_out)
                if len(model_outs) >= self.pairwise_batch_size:
                    break
        assert len(model_outs) == self.pairwise_batch_size

        model_outs = torch.cat(model_outs).to(self.device)
        logranks = torch.Tensor(logranks).to(self.device).float()

        loss = self.loss_fn(model_outs, logranks, w=(1-logranks))
        # TODO: proper loss magnitude normalization
        # loss_backprop = loss * (self.pairwise_batch_size**2)
        # loss_backprop = loss_backprop / (2 * self.batch_size * self.pairwise_batch_size - self.batch_size**2)
        loss.backward()

        self.opt.step()

        accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_outs)

        self.log_scalar('batch_loss_train', loss)
        self.log_scalar('batch_accuracy_train', accuracy)
        self.writer.add_histogram('batch_model_out_train', model_outs, self.step_ctr)
        self.writer.add_histogram('batch_model_target_train', logranks, self.step_ctr)

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
        self.opt.zero_grad()
        self.net.eval()

        with torch.no_grad():
            # accumulators
            model_outs, logranks = [], []

            for batch in dataset:
                if approx and len(model_outs) >= 2000:
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

        return float(accuracy.cpu().detach().numpy())


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

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
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

            return float(accuracy.detach().numpy())

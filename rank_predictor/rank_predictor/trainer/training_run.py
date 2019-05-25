import multiprocessing
import os
from typing import Dict, Callable, Union, List, Optional
import sacred
import torch
import numpy as np
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from graph_nets.data_structures.graph import Graph
from rank_predictor.data.v2.pagerank_dataset_screenshots import DatasetV2Screenshots
from rank_predictor.trainer.logging import setup_custom_logger
from rank_predictor.trainer.lr_scheduler.warmup_scheduler import GradualWarmupScheduler
from rank_predictor.trainer.ranking.utils import compute_batch_accuracy, compute_multi_batch_accuracy
from rank_predictor.data import threefold

logger = setup_custom_logger('run')


class TrainingRun:

    lr_scheduler_update_steps = 100

    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn: Callable,
                 data: threefold.Data, batch_size: int, device: torch.device, collate_fn: Callable = None,
                 lr_scheduler: Optional[GradualWarmupScheduler] = None) -> None:
        self.ex = ex
        self.name = name
        self.net = net
        self.opt = opt
        self.loss_fn = loss_fn
        self.step_ctr = 0
        self.device = device
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler

        save_dir = os.path.expanduser(os.environ['model_save_dir'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir

        cpu_count = multiprocessing.cpu_count()
        worker_count = max(cpu_count - 1, 1)
        worker_count = 0
        logger.info("Using {} workers for the data pipeline".format(worker_count))

        self.data = data

        # create data loader from dataset
        self.data_loader: threefold.Data[DataLoader] = threefold.Data(
            train=DataLoader(data.train, batch_size, shuffle=True, num_workers=worker_count, collate_fn=collate_fn),
            valid=DataLoader(data.valid, batch_size, shuffle=True, num_workers=worker_count, collate_fn=collate_fn),
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

            logger.info("Starting epoch #{}".format(epoch + 1))
            for batch in self.data_loader.train:
                if self.step_ctr % 5000 == 0:
                    logger.info("Running approx. validation at step #{}".format(self.step_ctr))
                    self._run_valid(self.data_loader.valid, 'valid', approx=True)
                    self._run_valid(self.data_loader.train, 'train', approx=True)
                    logger.info("Resuming training")

                self.step_ctr += 1
                try:
                    self._train_step(batch)
                except AssertionError as error:
                    logger.info("Skipping train step because of an error")
                    logger.info(error)
                    self.step_ctr -= 1
                self._lr_scheduler_update()
        self._save_model(epochs)

        logger.info("Training completed, reporting final accuracy")
        test_acc = self._run_valid(self.data_loader.test, 'test', approx=False)
        logger.info("Test acc.: {:.4f}".format(test_acc))
        valid_acc = self._run_valid(self.data_loader.valid, 'valid', approx=False)
        logger.info("Valid acc.: {:.4f}".format(valid_acc))
        return test_acc

    def _lr_scheduler_update(self) -> None:
        if self.step_ctr % self.lr_scheduler_update_steps != 0:
            return

        lr = self.opt.param_groups[0]['lr']
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch=self.step_ctr // self.lr_scheduler_update_steps)  # update learning rate
            lr = self.opt.param_groups[0]['lr']
            logger.info("Updating learning rate, now {}".format(lr))

        self.log_scalar('learning_rate', lr)

    def _save_model(self, epoch: int) -> None:
        save_dir = self.save_dir
        file_name = '{}_{:04d}.pt'.format(self.name, epoch)
        save_path = os.path.join(save_dir, file_name)

        logger.info("Storing model at '{}'.".format(save_path))

        torch.save(self.net.state_dict(), save_path)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
        raise NotImplementedError

    def log_scalar(self, name: str, val: Union[np.ndarray, torch.Tensor, int, float]):
        # tensorboard logging
        self.writer.add_scalar(name, val, self.step_ctr)

        # sacred logging
        if isinstance(val, torch.Tensor):
            val = float(val.cpu().detach().numpy())
        self.ex.log_scalar(name, val, self.step_ctr)


class GNTrainingRun(TrainingRun):

    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn, data: threefold.Data,
                 batch_size: int, pairwise_batch_size: int, device,
                 lr_scheduler: Optional[GradualWarmupScheduler] = None) -> None:
        def collate_fn(batch):
            return batch

        super().__init__(ex, name, net, opt, loss_fn, data, batch_size, device, collate_fn, lr_scheduler)

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


class FeatureExtractorTrainingRun(TrainingRun):

    def __init__(self, ex: sacred.Experiment, name: str, net: nn.Module, opt: optim.Adam, loss_fn: Callable,
                 data: threefold.Data, batch_size: int, device: torch.device,
                 lr_scheduler: Optional[GradualWarmupScheduler] = None) -> None:
        super().__init__(ex, name, net, opt, loss_fn, data, batch_size, device,
                         collate_fn=DatasetV2Screenshots.collate_fn, lr_scheduler=lr_scheduler)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> None:
        self.net.train()
        self.opt.zero_grad()

        desktop_imgs = batch['desktop_imgs'].to(self.device)
        mobile_imgs = batch['mobile_imgs'].to(self.device)
        logranks = batch['logranks'].to(self.device).float()
        weighting = batch['weighting'].to(self.device).float()

        model_out: torch.Tensor = self.net.forward(desktop_imgs, mobile_imgs)
        w = weighting * (1-logranks)
        loss = self.loss_fn(model_out, logranks, w=w)
        loss.backward()

        self.opt.step()

        accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_out)

        self.log_scalar('batch_loss_train', loss)
        self.log_scalar('batch_accuracy_train', accuracy)

    @staticmethod
    def remove_rank_duplicates_from_batch(batch) -> Dict:
        desktop_imgs, mobile_imgs, logranks = [], [], []
        prev_rank = None
        i = -1
        for rank in batch['ranks'].detach():
            i += 1
            if prev_rank is not None and rank == prev_rank:
                continue
            prev_rank = rank
            desktop_imgs.append(batch['desktop_imgs'][i])
            mobile_imgs.append(batch['mobile_imgs'][i])
            logranks.append(batch['logranks'][i])

        del batch['ranks']
        del batch['weighting']

        batch['desktop_imgs'] = torch.stack(desktop_imgs)
        batch['mobile_imgs'] = torch.stack(mobile_imgs)
        batch['logranks'] = torch.stack(logranks)

        return batch

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
        self.net.eval()

        with torch.no_grad():
            # accumulators
            loss_sum, model_out_batches, rank_batches = 0., [], []

            for batch in dataset:
                if approx and len(model_out_batches) >= 400:
                    break
                
                batch = self.remove_rank_duplicates_from_batch(batch)

                desktop_imgs = batch['desktop_imgs'].to(self.device)
                mobile_imgs = batch['mobile_imgs'].to(self.device)
                logranks = batch['logranks'].to(self.device).float()

                rank_batches.append(logranks)

                # forward pass
                model_out: torch.Tensor = self.net.forward(desktop_imgs, mobile_imgs)
                model_out_batches.append(model_out)

                loss = self.loss_fn(model_out, logranks, w=(1-logranks))
                loss_sum += loss

            n = len(model_out_batches)
            loss = loss_sum / n

            accuracy, _ = compute_multi_batch_accuracy(rank_batches, model_out_batches)

            self.log_scalar('loss_{}'.format(name), loss)
            self.log_scalar('accuracy_{}'.format(name), accuracy)

            return float(accuracy.detach().cpu().numpy())


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

    def _run_valid(self, dataset: Dataset, name: str, approx: bool = False) -> float:
        self.net.eval()

        with torch.no_grad():
            # accumulators
            loss_sum, model_out_batches, rank_batches = 0., [], []

            for batch in dataset:
                if approx and len(model_out_batches) >= 500:
                    break

                imgs: torch.Tensor = batch['img'].to(self.device)
                logranks: torch.Tensor = batch['logrank'].to(self.device).float()
                rank_batches.append(logranks)

                # forward pass
                model_out: torch.Tensor = self.net.forward(imgs)
                model_out_batches.append(model_out)

                loss = self.loss_fn(model_out, logranks, w=(1-logranks))
                loss_sum += loss

            n = len(model_out_batches)
            loss = loss_sum / n

            accuracy, _ = compute_multi_batch_accuracy(rank_batches, model_out_batches)

            self.log_scalar('loss_{}'.format(name), loss)
            self.log_scalar('accuracy_{}'.format(name), accuracy)

            return float(accuracy.detach().cpu().numpy())

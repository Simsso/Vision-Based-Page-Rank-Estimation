import logging
import os

from sacred.observers import MongoObserver
from sacred.stflow import LogFileWriter

from rank_predictor.data.v2.pagerank_dataset import DatasetV2
import torch
from rank_predictor.model.graph_baseline import GraphBaseline
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from rank_predictor.trainer.training_run import GNTrainingRun
from sacred import Experiment

name = 'v2/test_training2'
ex = Experiment(name)

ex.observers.append(MongoObserver.create(url='mongodb://localhost:27017/sacred'))


@ex.config
def run_config():
    learning_rate: float = 4e-4
    batch_size = 3
    epochs = 10
    optimizer = 'adam'


@ex.main
def train(learning_rate: float, batch_size: int, epochs: int, optimizer: str):
    logging.basicConfig(level=logging.INFO)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # dataset
    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    data = DatasetV2.get_threefold(dataset_dir, train_ratio=0.85, valid_ratio=0.1)

    # model with weights
    net = GraphBaseline()
    if device == 'cuda':
        net.cuda()

    # optimizer
    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer '{}'".format(optimizer))

    # loss
    loss = ProbabilisticLoss()

    training_run = GNTrainingRun(ex, name, net, opt, loss, data, batch_size, device)
    training_run(epochs)


if __name__ == '__main__':
    ex.run_commandline()

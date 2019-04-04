import logging
import os
from sacred.observers import MongoObserver
from torch import nn

from rank_predictor.data.v2.pagerank_dataset import DatasetV2
import torch
from rank_predictor.model.graph_baseline import GraphBaseline
from rank_predictor.model.graph_connected import GraphConnected
from rank_predictor.model.graph_extractor_full import GraphExtractorFull
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from rank_predictor.trainer.training_run import GNTrainingRun
from sacred import Experiment

name = 'fullc_12_2c'
ex = Experiment(name)

ex.observers.append(MongoObserver.create(url='mongodb://localhost:27017/sacred'))


@ex.config
def run_config():
    learning_rate: float = 1e-4
    batch_size = 2
    pairwise_batch_size = 2
    epochs = 10
    optimizer = 'adam'
    train_ratio, valid_ratio = .85, .1
    model_name = 'GraphExtractorFull'
    loss = 'ProbabilisticLoss'
    weighting = 'c_ij = c_ij'
    logrank_b = 1.5
    drop_p = .05
    num_core_blocks = 2


@ex.main
def train(learning_rate: float, batch_size: int, pairwise_batch_size: int, epochs: int, optimizer: str,
          train_ratio: float, valid_ratio: float, model_name: str, loss: str, logrank_b: float, drop_p: float,
          num_core_blocks: int) -> str:
    logging.basicConfig(level=logging.INFO)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # dataset
    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    data = DatasetV2.get_threefold(dataset_dir, train_ratio, valid_ratio, logrank_b)

    # model with weights
    if model_name == 'GraphBaseline':
        net = GraphBaseline()
    elif model_name == 'GraphConnected':
        net = GraphConnected()
    elif model_name == 'GraphExtractorFull':
        net = GraphExtractorFull(num_core_blocks, drop_p)
    else:
        raise ValueError("Unknown model name '{}'".format(model_name))

    if device == 'cuda':
        net.cuda()

    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer '{}'".format(optimizer))

    if loss == 'ProbabilisticLoss':
        loss = ProbabilisticLoss()
    else:
        raise ValueError("Unknown loss '{}'".format(loss))

    training_run = GNTrainingRun(ex, name, net, opt, loss, data, batch_size, pairwise_batch_size, device)
    val_acc = training_run(epochs)

    return "Val acc: {:.4f}".format(val_acc)


if __name__ == '__main__':
    ex.run_commandline()

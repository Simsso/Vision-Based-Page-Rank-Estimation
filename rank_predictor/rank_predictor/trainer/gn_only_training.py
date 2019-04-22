import logging
import os
import torch
from rank_predictor.data.v2.pagerank_dataset_cached import DatasetV2Cached
from rank_predictor.model.graph_only_models import GNAvg, GNMax, GNDeep
from rank_predictor.trainer.training_run import GNTrainingRun
from rank_predictor.trainer.lr_scheduler.warmup_scheduler import GradualWarmupScheduler
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from sacred import Experiment
from sacred.observers import MongoObserver

name = 'gn_only_fe_08_deep_01'
ex = Experiment(name)

ex.observers.append(MongoObserver.create(url='mongodb://localhost:27017/sacred'))


@ex.config
def run_config():
    learning_rate: float = 5e-6
    batch_size = 8
    pairwise_batch_size = 8
    epochs = 20
    optimizer = 'adam'
    train_ratio, valid_ratio = .6, .2
    model_name = 'GNDeep'
    loss = 'ProbabilisticLoss'
    weighting = 'c_ij = c_ij'
    logrank_b = 10
    drop_p = 0.1
    num_core_blocks = 1
    share_core_weights = False
    lr_scheduler = 'None'
    lr_scheduler_gamma = None
    feat_extr_weights_path = os.path.expanduser('~/dev/pagerank/models/featextr_08_0010.pt')


@ex.main
def train(learning_rate: float, batch_size: int, pairwise_batch_size: int, epochs: int, optimizer: str,
          train_ratio: float, valid_ratio: float, model_name: str, loss: str, logrank_b: float, drop_p: float,
          num_core_blocks: int, lr_scheduler: str, lr_scheduler_gamma: float, feat_extr_weights_path: str,
          share_core_weights: bool) -> str:
    logging.basicConfig(level=logging.INFO)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # dataset
    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    data = DatasetV2Cached.get_threefold(dataset_dir, train_ratio, valid_ratio, logrank_b,
                                         feat_extr_weights_path=feat_extr_weights_path)

    # model with weights
    if model_name == 'GNAvg':
        net = GNAvg()
    elif model_name == 'GNMax':
        net = GNMax()
    elif model_name == 'GNDeep':
        net = GNDeep(drop_p, num_core_blocks, shared_weights=share_core_weights)
    else:
        raise ValueError("Unknown model name '{}'".format(model_name))

    if device.type == 'cuda':
        net.cuda()

    if optimizer == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer '{}'".format(optimizer))

    if loss == 'ProbabilisticLoss':
        loss = ProbabilisticLoss()
    else:
        raise ValueError("Unknown loss '{}'".format(loss))

    if lr_scheduler == 'GradualWarmupSchedulerExponentialLR':
        exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_scheduler_gamma)
        lr_scheduler = GradualWarmupScheduler(opt, multiplier=15, total_epoch=20, after_scheduler=exp_scheduler)
    else:
        lr_scheduler = None

    training_run = GNTrainingRun(ex, name, net, opt, loss, data, batch_size, pairwise_batch_size, device, lr_scheduler)
    val_acc = training_run(epochs)

    return "Val acc: {:.4f}".format(val_acc)


if __name__ == '__main__':
    ex.run_commandline()

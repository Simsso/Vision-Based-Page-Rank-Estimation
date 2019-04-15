import logging
import os
import torch
from rank_predictor.model.graph_extractor_full import ScreenshotsFeatureExtractor
from rank_predictor.model.graph_only_models import GNAvg
from rank_predictor.trainer.training_run import GNTrainingRun
from rank_predictor.trainer.lr_scheduler.warmup_scheduler import GradualWarmupScheduler
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from sacred import Experiment
from sacred.observers import MongoObserver

name = 'gn_only_test'
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
    model_name = 'GNAvg'
    loss = 'ProbabilisticLoss'
    weighting = 'c_ij = c_ij'
    logrank_b = 1.5
    drop_p = .05
    num_core_blocks = 2
    lr_scheduler = 'None'
    lr_scheduler_gamma = 0.98
    feat_extr_weights_path = os.path.expanduser('~/dev/pagerank/models/featextr_04_0007.pt')


@ex.main
def train(learning_rate: float, batch_size: int, pairwise_batch_size: int, epochs: int, optimizer: str,
          train_ratio: float, valid_ratio: float, model_name: str, loss: str, logrank_b: float, drop_p: float,
          num_core_blocks: int, lr_scheduler: str, lr_scheduler_gamma: float, feat_extr_weights_path: str) -> str:
    logging.basicConfig(level=logging.INFO)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # dataset
    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    data = DatasetV2.get_threefold(dataset_dir, train_ratio, valid_ratio, logrank_b)

    # restore feature extractor
    logging.info("Restoring pre-trained model weights")
    feat_extr = ScreenshotsFeatureExtractor(drop_p=0)
    feat_extr.load_state_dict(torch.load(feat_extr_weights_path))

    # model with weights
    if model_name == 'GNAvg':
        net = GNAvg(feat_extr)
    else:
        raise ValueError("Unknown model name '{}'".format(model_name))

    if device.type == 'cuda':
        feat_extr.cuda()
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

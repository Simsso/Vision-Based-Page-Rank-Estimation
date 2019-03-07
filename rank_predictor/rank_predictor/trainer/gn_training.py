import logging
import os
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
import torch

from rank_predictor.model.graph_baseline import GraphBaseline
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from rank_predictor.trainer.training_run import TrainingRun

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
opt = torch.optim.Adam(net.parameters(), lr=1e-4)

# loss
loss = ProbabilisticLoss()

training_run = TrainingRun(net, opt, loss, data, batch_size=1, device=device)
training_run(50)

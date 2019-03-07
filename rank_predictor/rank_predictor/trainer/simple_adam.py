import logging
import os
import torch
from rank_predictor.data.v1.pagerank_dataset import DatasetV1
from rank_predictor.model.screenshot_feature_extractor import DesktopScreenshotFeatureExtractor
from torch import optim
from rank_predictor.trainer.training_run import TrainingRun
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss

logging.basicConfig(level=logging.INFO)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# dataset
dataset_dir = os.path.expanduser(os.getenv('dataset_v1_path'))
dataset_v1 = DatasetV1.get_threefold(dataset_dir, train_ratio=0.85, valid_ratio=0.1)

# model with weights
net = DesktopScreenshotFeatureExtractor()
net.cuda()

# optimizer
opt = optim.Adam(net.parameters(), lr=2e-4)

# loss
loss = ProbabilisticLoss()

training_run = TrainingRun(net, opt, loss, dataset_v1, batch_size=24, device=device)
training_run(50)

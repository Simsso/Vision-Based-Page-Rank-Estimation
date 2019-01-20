import logging
import os
import torch
from rank_predictor.data.v1.pagerank_dataset import ScreenshotPagerankDatasetV1
from rank_predictor.model.screenshot_feature_extractor import ScreenshotFeatureExtractor
from torch import nn, optim
from rank_predictor.trainer.training_run import TrainingRun


logging.basicConfig(level=logging.INFO)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# dataset
dataset_dir = os.getenv('dataset_v1_path')
dataset_v1 = ScreenshotPagerankDatasetV1.get_threefold(dataset_dir, train_ratio=0.85, valid_ratio=0.1)

# model with weights
net = ScreenshotFeatureExtractor()
net.cuda()

# optimizer
opt = optim.Adam(net.parameters(), lr=1e-5)

# loss
train_label_weights = dataset_v1.train.get_label_weights()
train_label_weights = torch.Tensor(train_label_weights).to(device)
logging.info("Using label weights: {}".format(str(train_label_weights)))
loss = nn.CrossEntropyLoss(train_label_weights)

training_run = TrainingRun(net, opt, loss, dataset_v1, batch_size=24, device=device)
training_run(1)

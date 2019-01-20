import os

import torch

from rank_predictor.data.v1.pagerank_dataset import ScreenshotPagerankDatasetV1
from rank_predictor.data.v1.transforms import ImageTransform, ToCudaTensor
from rank_predictor.model.screenshot_feature_extractor import ScreenshotFeatureExtractor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, ToPILImage

from rank_predictor.trainer.training_run import TrainingRun

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

composed_transforms = Compose([
    ImageTransform(ToPILImage()),
    ImageTransform(Resize((1920//4, 1080//4))),
    ImageTransform(ToCudaTensor(device)),
    ImageTransform(Normalize((.5, .5, .5, .5), (.5, .5, .5, .5))),
])

# dataset
pagerank_v1 = ScreenshotPagerankDatasetV1(
    root_dir=os.getenv('dataset_v1_path'), transform=composed_transforms)

# dataset loader
dataloader = DataLoader(pagerank_v1, batch_size=4, shuffle=True)

# model with weights
net = ScreenshotFeatureExtractor()
net.cuda()

# optimizer
opt = optim.Adam(net.parameters(), lr=3e-5)

training_run = TrainingRun(net, opt, nn.MSELoss(), dataloader, device)
training_run(1)

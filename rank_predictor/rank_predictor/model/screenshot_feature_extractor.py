from __future__ import print_function
import rank_predictor.model.util as uf
import torch.nn as nn
import torch.nn.functional as F


class ScreenshotFeatureExtractor(nn.Module):
    """
    Screenshot feature extraction architecture inspied by "pix2code".
    (arXiv link: https://arxiv.org/abs/1705.07962)
    """

    def __init__(self):
        super(ScreenshotFeatureExtractor, self).__init__()

        self.conv1a = nn.Conv2d(4, 32, kernel_size=(3, 3))
        self.conv1b = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.conv2a = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 3))

        self.conv3a = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv3b = nn.Conv2d(128, 128, kernel_size=(3, 3))

        self.dense1 = nn.Linear(1024, 256)  # diverging from the paper here
        self.dense2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.dropout2d(x, p=.25, training=self.training)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, (3, 3))  # originally stride (2, 2)
        x = F.dropout2d(x, p=.25, training=self.training)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, (3, 3))  # originally stride (2, 2)
        x = F.avg_pool2d(x, (5, 5))  # originally not included
        x = F.dropout2d(x, p=.25, training=self.training)

        x = uf.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=.3, training=self.training)

        x = self.dense2(x)
        x = F.softmax(x, dim=1)

        return x

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScreenshotFeatureExtractor(nn.Module):
    """
    Screenshot feature extraction architecture inspied by "pix2code".
    (arXiv link: https://arxiv.org/abs/1705.07962)
    """

    def __init__(self):
        super(ScreenshotFeatureExtractor, self).__init__()

        self.conv1a = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.conv1b = nn.Conv2d(32, 32, kernel_size=(3, 3))

        self.conv2a = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv2b = nn.Conv2d(64, 64, kernel_size=(3, 3))

        self.conv3a = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.conv3b = nn.Conv2d(128, 128, kernel_size=(3, 3))

        self.dense1 = nn.Linear(128*5*5, 1024)
        self.dense2 = nn.Linear(1024, 1024)

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, (2, 2))
        x = F.dropout2d(x, p=.25, training=True)  # TODO: training is hard-coded

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, (3, 3))  # originally stride (2, 2)
        x = F.dropout2d(x, p=.25)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, (3, 3))  # originally stride (2, 2)
        x = F.avg_pool2d(x, (10, 10))  # originally not included
        x = F.dropout2d(x, p=.25)

        x = x.view(-1, self._num_flat_features(x))
        x = F.relu(self.dense1(x))
        x = F.dropout(x, p=.3)

        x = F.relu(self.dense2(x))
        x = F.dropout(x, p=.3)

        return x

    @staticmethod
    def _num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

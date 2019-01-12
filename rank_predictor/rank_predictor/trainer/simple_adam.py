import matplotlib.pyplot as plt
import numpy as np
import os
from rank_predictor.data.v1.pagerank_dataset import ScreenshotPagerankDatasetV1
from rank_predictor.data.v1.transforms import ImageTransform
from rank_predictor.model.screenshot_feature_extractor import ScreenshotFeatureExtractor
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor, Compose, ToPILImage

composed_transforms = Compose([
    ImageTransform(ToPILImage()),
    ImageTransform(ToTensor()),
])

# dataset
pagerank_v1 = ScreenshotPagerankDatasetV1(
    root_dir=os.getenv('dataset_v1_path'), transform=composed_transforms)

# dataset loader
dataloader = DataLoader(pagerank_v1, batch_size=32, shuffle=True)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


for batch in dataloader:
    imgs = batch['img']
    ranks = batch['rank']
    imshow(imgs[0])

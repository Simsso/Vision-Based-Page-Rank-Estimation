import os
from skimage import io
import torch
from torch.utils.data import Dataset
from typing import Dict


class ScreenshotPagerankDatasetV1(Dataset):
    """
    Dataset v1 of screenshot pageranks.
    Each sample is a dictionary with
     * 'img' being a C x H x W tensor
     * 'rank' being a tensor indicating the page rank
    """

    def __init__(self, root_dir: str) -> None:
        super(ScreenshotPagerankDatasetV1).__init__()

        if not os.path.isdir(root_dir):
            raise ValueError("The provided path '{}' is not a directory".format(root_dir))
        self.root_dir = root_dir
        self.img_names = os.listdir(self.root_dir)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        img_name = self.img_names[index]
        img_path = os.path.join(self.root_dir, img_name)
        img = io.imread(img_path)
        # convert to channels x height x width
        img = img.transpose((2, 0, 1))

        rank = int(os.path.splitext(img_name)[0])

        return {
            'img': torch.Tensor(img),
            'rank': torch.Tensor(rank)
        }

    def __len__(self) -> int:
        return len(self.img_names)

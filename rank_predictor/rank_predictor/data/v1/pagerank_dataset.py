import glob
import os

import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from torchvision.transforms import ToPILImage, Resize, Normalize, Compose, ToTensor
from tqdm import tqdm
from math import log
from rank_predictor.data.v1.utils import img_loading_possible
from rank_predictor.data import threefold
from rank_predictor.data.v1.transforms import ImageTransform, ToCudaTensor
import logging


class DatasetV1(Dataset):
    """
    Dataset v1 of screenshot pageranks.
    Each sample is a dictionary with
     * 'img' being a C x H x W tensor (after applying a ToTensor transformation)
     * 'rank' being a tensor indicating the page rank
     * 'label' being a "class" that the image belongs to (derived from its rank)
     * 'logrank' being the logarithmically scaled rank (e.g. 80,001 and 90,000 are closer together than 1 and 10,000)
    """

    num_labels = 3
    max_rank = 10**5

    def __init__(self, img_paths: List[str]) -> None:
        super(DatasetV1).__init__()

        self.img_paths = img_paths
        ranks = map(self.filename_to_rank, self.img_paths)
        self.labels = list(map(self.rank_to_label, ranks))

        self.transform = Compose([
            ImageTransform(ToPILImage()),
            ImageTransform(Resize((1920 // 4, 1080 // 4))),
            ImageTransform(ToTensor()),
            ImageTransform(Normalize((.5, .5, .5, .5), (.5, .5, .5, .5))),
        ])

    def __getitem__(self, index: int) -> Dict[str, any]:
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        # load the image
        try:
            img = io.imread(img_path)
        except ValueError:
            logging.error("The image '{}' could not be loaded.".format(img_path))
            raise

        # parse rank from image name
        rank = self.filename_to_rank(img_name)

        sample = {
            'img': img,
            'rank': rank,
            'label': self.rank_to_label(rank),
            'logrank': self.rank_to_logrank(rank),
        }

        # apply pre-processing
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        assert len(self.img_paths) == len(self.labels), "Labels must have equal length as images"
        
        return len(self.img_paths)

    def get_label_weights(self) -> List[int]:
        label_ctr = [0] * self.num_labels
        for label_val in self.labels:
            label_ctr[label_val] += 1

        n = len(label_ctr)
        label_weights = list(map(lambda c: (n / c), label_ctr))

        return label_weights

    @staticmethod
    def filename_to_rank(file_name: str) -> int:
        """
        Converts a filename into the corresponding rank, e.g. "1234.jpg" --> 1234 (integer)
        :param file_name: File name, e.g. "1234.jpg"
        :return: Rank for the file, e.g. 1234
        """
        try:
            path_parts = os.path.split(file_name)
            filename_with_ext = path_parts[-1]
            rank = int(os.path.splitext(filename_with_ext)[0])
        except ValueError:
            logging.error("Could not parse the file name '{}'.".format(file_name))
            raise
        return rank
    
    @staticmethod
    def rank_to_label(rank: int) -> int:
        assert 0 < rank <= DatasetV1.max_rank, "Rank '{}' is out of range.".format(rank)

        if rank <= 10**3:
            return 0
        if rank <= 10**4:
            return 1
        return 2

    @staticmethod
    def rank_to_logrank(rank: int, b: float = 1.) -> float:
        """
        Maps a rank from {1, ..., max_rank} to [0,1] in a logarithmic fashion.
        :param rank: The rank to map, in {1, ..., max_rank}
        :param b: Base makes the weighting steeper b --> 0, more linear b --> 10, or inverted b > 10
        :return: Scalar in [0,1]
        """

        max_rank = DatasetV1.max_rank
        assert 0 < rank <= max_rank, "Rank '{}' is out of range.".format(rank)

        return pow(log(rank*max_rank) / log(max_rank) - 1., b)

    @staticmethod
    def from_path(root_dir: str):
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        img_paths = sorted(glob.glob(os.path.join(root_dir, '*.png')))

        return DatasetV1(img_paths)

    @staticmethod
    def get_threefold(root_dir: str, train_ratio: float, valid_ratio: float) -> threefold.Data:
        """
        Load dataset from root_dir and split it into three parts (train, validation, test).
        The function splits in a deterministic way.
        :param root_dir: Directory of the dataset
        :param train_ratio: Value in [0,1] defining the ratio of training samples
        :param valid_ratio: Value in [0,1] defining the ratio of validation samples
        :return: Three datasets (train, validation, test)
        """

        assert train_ratio + valid_ratio <= 1., "Train and validation ratio must be less than or equal to 1."
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        logging.info("Loading and splitting dataset v1")

        img_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))

        assert len(img_paths) > 0, "No images found at '{}'".format(root_dir)

        train_paths, valid_paths, test_paths = [], [], []

        for path in tqdm(img_paths):
            if os.getenv('validate_imgs', False) and not img_loading_possible(path):
                logging.warning("The image '{}' could not be loaded".format(path))
                continue

            n_train, n_valid, n_test = len(train_paths), len(valid_paths), len(test_paths)
            n_total = n_train + n_valid + n_test

            if n_total == 0 or n_train / n_total < train_ratio:
                train_paths.append(path)
            elif n_valid / n_total < valid_ratio:
                valid_paths.append(path)
            else:
                test_paths.append(path)

        return threefold.Data(
            train=DatasetV1(train_paths),
            valid=DatasetV1(valid_paths),
            test=DatasetV1(test_paths))

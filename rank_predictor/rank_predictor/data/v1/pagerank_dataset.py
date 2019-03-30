from glob import glob
import os
from torch.utils.data import Dataset
from typing import Dict, List
from torchvision.transforms import ToPILImage, Resize, Normalize, Compose, ToTensor
from rank_predictor.data.threefold import get_threefold
from rank_predictor.data.utils import filename_to_rank, load_image, rank_to_logrank
from rank_predictor.data import threefold
from rank_predictor.data.v1.transforms import ImageTransform


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

    def __init__(self, img_paths: List[str], logrank_b: float) -> None:
        super(DatasetV1).__init__()

        self.img_paths = img_paths
        ranks = map(filename_to_rank, self.img_paths)
        self.labels = list(map(self.rank_to_label, ranks))

        self.transform = Compose([
            ImageTransform(ToPILImage()),
            ImageTransform(Resize((1920 // 4, 1080 // 4))),
            ImageTransform(ToTensor()),
            ImageTransform(Normalize((.5, .5, .5, .5), (.5, .5, .5, .5))),
        ])

        self.logrank_b = logrank_b

    def __getitem__(self, index: int) -> Dict[str, any]:
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)

        img = load_image(img_path)

        # parse rank from image name
        rank = filename_to_rank(img_name)

        sample = {
            'img': img,
            'rank': rank,
            'label': self.rank_to_label(rank),
            'logrank': rank_to_logrank(DatasetV1.max_rank, rank, self.logrank_b),
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
    def rank_to_label(rank: int) -> int:
        assert 0 < rank <= DatasetV1.max_rank, "Rank '{}' is out of range.".format(rank)

        if rank <= 10**3:
            return 0
        if rank <= 10**4:
            return 1
        return 2

    @staticmethod
    def from_path(root_dir: str):
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        img_paths = sorted(glob(os.path.join(root_dir, '*.png')))

        return DatasetV1(img_paths)

    @staticmethod
    def get_threefold(root_dir: str, train_ratio: float, valid_ratio: float, logrank_b: float) -> threefold.Data:
        """
        Load dataset from root_dir and split it into three parts (train, validation, test).
        The function splits in a deterministic way.
        :param root_dir: Directory of the dataset
        :param train_ratio: Value in [0,1] defining the ratio of training samples
        :param valid_ratio: Value in [0,1] defining the ratio of validation samples
        :param logrank_b: Logrank base (makes the weighting steeper b --> 0, more linear b --> 10, or inverted b > 10)
        :return: Three datasets (train, validation, test)
        """
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        sample_paths = glob(os.path.join(root_dir, '*.jpg'))

        return get_threefold(DatasetV1, sample_paths, train_ratio, valid_ratio, logrank_b)

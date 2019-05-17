import os
from typing import Set

import torch
from torch.utils.data import Dataset

from rank_predictor.data import threefold
from rank_predictor.data.threefold import get_threefold
from rank_predictor.data.v2.pagerank_dataset import DatasetV2


class DatasetV2Screenshots(Dataset):
    """
    Dataset v2 of website screenshots (w/o graph) with associated rank.
    Each sample is a dictionary with
     * 'rank' being a tensor indicating the page rank
     * 'mobile_imgs' mobile screenshots (up to 8)
     * 'desktop_imgs' desktop screenshots (up to 8)
     * 'logrank' being the logarithmically scaled rank (e.g. 80,001 and 90,000 are closer together than 1 and 10,000)
    """

    def __init__(self, page_paths: Set[str], logrank_b: float) -> None:
        super().__init__()

        self.graph_dataset = DatasetV2(page_paths, logrank_b)

    def __getitem__(self, index):
        sample = self.graph_dataset[index]
        g = sample['graph']
        desktop_imgs, mobile_imgs = [], []

        for node in g.nodes:
            desktop_imgs.append(node.attr.val['desktop_img'])
            mobile_imgs.append(node.attr.val['mobile_img'])

        assert len(mobile_imgs) == len(desktop_imgs)

        desktop_imgs = torch.stack(desktop_imgs)
        mobile_imgs = torch.stack(mobile_imgs)

        sample['desktop_imgs'] = desktop_imgs
        sample['mobile_imgs'] = mobile_imgs

        del sample['graph']

        return sample

    def __len__(self):
        return len(self.graph_dataset)

    @staticmethod
    def collate_fn(samples):
        """
        Pass this function to the data loader (PyTorch class DataLoader) when using this dataset.
        :param samples: Sample as returned from self.__getitem__
        :return: Samples with updated weighting accounting for the variable number of screenshots per graph.
        """
        desktop_imgs, mobile_imgs = [], []
        ranks, logranks = [], []
        weighting = []

        for sample in samples:
            n = sample['desktop_imgs'].size(0)
            desktop_imgs.append(sample['desktop_imgs'])
            mobile_imgs.append(sample['mobile_imgs'])
            rank = sample['rank']
            logrank = sample['logrank']
            ranks.extend([rank] * n)
            logranks.extend([logrank] * n)
            weighting.extend([1/n] * n)

        return {
            'desktop_imgs': torch.cat(desktop_imgs),
            'mobile_imgs': torch.cat(mobile_imgs),
            'ranks': torch.Tensor(ranks),
            'logranks': torch.Tensor(logranks),
            'weighting': torch.Tensor(weighting)
        }

    @staticmethod
    def from_path(root_dir: str, logrank_b: float):
        page_paths = DatasetV2.get_page_paths(root_dir)
        return DatasetV2Screenshots(page_paths, logrank_b)

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

        page_paths = list(DatasetV2.get_page_paths(root_dir))
        return get_threefold(DatasetV2Screenshots, page_paths, train_ratio, valid_ratio, logrank_b)
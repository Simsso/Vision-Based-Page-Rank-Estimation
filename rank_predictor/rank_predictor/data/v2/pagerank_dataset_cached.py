import logging
import os
from glob import glob
from typing import Set, Dict
import torch
from tqdm import tqdm

from rank_predictor.data.utils import folder_to_rank
from rank_predictor.data.threefold import get_threefold
from graph_nets.data_structures.graph import Graph
from rank_predictor.data import threefold
from rank_predictor.data.v2.pagerank_dataset import DatasetV2


class DatasetV2Cached(DatasetV2):

    graph_cache = None

    def __init__(self, page_paths: Set[str], logrank_b: float, feat_extr_weights_path: str) -> None:
        super().__init__(page_paths, logrank_b)
        if DatasetV2Cached.graph_cache is None:
            DatasetV2Cached.graph_cache = self.load_cached_graphs(feat_extr_weights_path)

        logging.info("Validating cached dataset entries")
        for page_path in tqdm(page_paths):
            rank = folder_to_rank(page_path)
            assert rank in DatasetV2Cached.graph_cache, "Could not find rank #{} in the graph cache".format(rank)

    def load_graph(self, path: str) -> Graph:
        """
        Overwritten because instead of loading a JSON file this method returns a graph from the cache.
        :return: Graph from the cache where node attributes are e.g. just feature vectors.
        """
        rank = folder_to_rank(path)
        assert rank in DatasetV2Cached.graph_cache, "Could not find rank #{} in the graph cache".format(rank)
        return DatasetV2Cached.graph_cache[rank]

    @staticmethod
    def load_cached_graphs(feat_extr_weights_path: str) -> Dict[int, Graph]:
        """
        Loads several graph cache dictionaries, merges them into one and returns it.
        The dicts may be created with the cache_activations.py script.
        :param feat_extr_weights_path: Path to the model file, e.g. 'file/featextr_08_0010.pt'.
                                       The cached files are in 'file/featextr_08_0010.pt-cache-*.pt'
        :return: Dict from rank (int) to graph object.
        """
        cache_paths = glob('{}-cache-*.pt'.format(feat_extr_weights_path))
        logging.info("Loading cached graphs from #{} files".format(len(cache_paths)))

        cache, entry_count = {}, 0

        for cache_path in tqdm(cache_paths):
            file_cache = torch.load(cache_path)
            cache.update(file_cache)

            entry_count += len(file_cache)

        # convert keys from string to int
        cache = {int(k): cache[k] for k in cache}

        assert entry_count == len(cache), "Found duplicate entries in the files"
        logging.info("Found #{} graphs".format(entry_count))

        return cache

    @staticmethod
    def get_threefold(root_dir: str, train_ratio: float, valid_ratio: float, logrank_b: float, **kwargs)\
            -> threefold.Data:
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
        return get_threefold(DatasetV2Cached, page_paths, train_ratio, valid_ratio, logrank_b, **kwargs)

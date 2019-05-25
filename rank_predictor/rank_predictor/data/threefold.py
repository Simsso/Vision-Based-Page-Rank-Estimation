import logging
from collections import namedtuple
from typing import List, Type

from tqdm import tqdm

Data = namedtuple('ThreefoldData', ['train', 'valid', 'test'])


def get_threefold(klass: Type, sample_paths: List[str], train_ratio: float, valid_ratio: float, logrank_b: float,
                  **kwargs) -> Data:
    """
    :param klass: Dataset class, e.g. DatasetV2
    :param sample_paths: List of paths that point to the samples of the dataset
    :param train_ratio: Value in [0,1] defining the ratio of training samples
    :param valid_ratio: Value in [0,1] defining the ratio of validation samples
    :param logrank_b: Logrank base (makes the weighting steeper b --> 0, more linear b --> 10, or inverted b > 10)
    :return: Three datasets (train, validation, test)
    """

    assert train_ratio + valid_ratio <= 1., "Train and validation ratio must be less than or equal to 1."
    assert len(sample_paths) > 0, "No dataset samples found."

    logging.info("Loading and splitting dataset")

    sample_paths = sorted(sample_paths)

    train_paths, valid_paths, test_paths = [], [], []

    for path in tqdm(sample_paths):
        n_train, n_valid, n_test = len(train_paths), len(valid_paths), len(test_paths)
        n_total = n_train + n_valid + n_test

        if n_total == 0 or n_train / n_total < train_ratio:
            train_paths.append(path)
        elif n_valid / n_total < valid_ratio:
            valid_paths.append(path)
        else:
            test_paths.append(path)

    return Data(
        train=klass(train_paths, logrank_b, **kwargs),
        valid=klass(valid_paths, logrank_b, **kwargs),
        test=klass(test_paths, logrank_b, **kwargs))

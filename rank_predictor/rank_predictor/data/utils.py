from math import log
from typing import List

from skimage import io
import logging
import numpy as np
import os

from rank_predictor.data import threefold

Image = np.ndarray


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


def folder_to_rank(folder_path: str) -> int:
    try:
        folders = folder_path.split(os.path.sep)
        folders: List[str] = list(filter(len, folders))  # remove empty entries, e.g. caused by "folder/123/"

        folder_name: str = folders[-1]

        rank = int(folder_name)
    except ValueError:
        logging.error("Could not parse the folder path '{}'.".format(folder_path))
        raise
    return rank


def load_image(path: str) -> np.ndarray:
    # load the image
    try:
        return io.imread(path)
    except ValueError:
        logging.error("The image '{}' could not be loaded.".format(img_path))
        raise


def rank_to_logrank(max_rank: int, rank: int, b: float = 1.) -> float:
    """
    Maps a rank from {1, ..., max_rank} to [0,1] in a logarithmic fashion.
    :param max_rank: The maximum value that `rank` may take
    :param rank: The rank to map, in {1, ..., max_rank}
    :param b: Base makes the weighting steeper b --> 0, more linear b --> 10, or inverted b > 10
    :return: Scalar in [0,1]
    """

    assert 0 < rank <= max_rank, "Rank '{}' is out of range.".format(rank)

    return pow(log(rank*max_rank) / log(max_rank) - 1., b)

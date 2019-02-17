from typing import List

from skimage import io
import logging
import numpy as np
import os

Image = np.ndarray


def filename_to_rank(file_name: str) -> int:
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

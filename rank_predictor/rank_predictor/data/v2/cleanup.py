"""
Dataset v2 clean-up script.
Removes all pages which have only images each less than x bytes in size.
"""

from glob import glob
import os
from typing import Set
from tqdm import tqdm
import shutil


root_dir = os.path.expanduser('~/Development/pagerank/data/v2')
assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

img_size_limit = 3500  # number of bytes

query_str = os.path.join(root_dir, '*', '')
page_paths = sorted(glob(query_str))
page_paths = set(page_paths)


def get_images(p_path: str) -> Set[str]:
    """
    :return: Set of image paths of the given page.
    """
    q_str = os.path.join(p_path, 'image', '*.jpg')
    img_paths = glob(q_str)
    img_paths = set(img_paths)
    return img_paths


to_delete = set()

tqdm.write('Searching for invalid pages...')

for page_path in tqdm(page_paths):
    all_below = True
    for img_path in get_images(page_path):
        size = os.path.getsize(img_path)
        if size > img_size_limit:
            all_below = False
            break
    if all_below:
        to_delete.add(page_path)

tqdm.write('Deleting {} folders'.format(len(to_delete)))
for page_path in to_delete:
    shutil.rmtree(page_path)

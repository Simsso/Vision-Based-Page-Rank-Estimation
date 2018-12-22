"""
Deprecated because the split now happens within PyTorch.

This script pre-processes raw data of dataset v1, i.e. a folder
of screenshots where each file is called xxx.png where xxx is its
ranking (integer).

The environment variable PATH points to the folder. The dataset is
being modified in-place. Modifications are the following.

 * removal of blank images (only transparent, black, or white)
 * conversion from PNG to JPG
 * random split into training, validation, and test set
"""

import os
from PIL import Image, ImageChops
import random

dataset_path = os.environ['PATH']
splits = {
    'train': 0.6,
    'validation': 0.2,
    'test': 0.2,
}

if not dataset_path:
    print("Environment variable 'PATH' must be set")
    exit(1)

print("Dataset directory '{}'".format(dataset_path))

if not os.path.isdir(dataset_path):
    print("Path does not point to a directory".format(dataset_path))
    exit(1)

files = os.listdir(dataset_path)
if not files:
    print("Did not find any files in the provided directory")

print("Found {} file(s)".format(str(len(files))))


def is_valid_img(path: str) -> bool:
    # check file extension
    if path[-4:] != '.png':
        return False

    # check alpha channel
    img = Image.open(path, 'r')
    has_alpha = img.mode == 'RGBA'
    if not has_alpha:
        return True
    if not img.getbbox() or not ImageChops.invert(img).getbbox():
        return False
    return True


def get_split_assignment() -> str:
    rand = random.uniform(0, 1)
    accum = 0.
    for assignment, p in splits.items():
        accum += p
        if accum >= rand:
            return assignment
    raise ValueError("Invalid assignment probabilities (they must add up to 1)")


# create split sub-folders
print("Creating sub-folders:")
print(splits)
for sub_folder in splits:
    try:
        os.mkdir(os.path.join(dataset_path, sub_folder))
    except FileExistsError:
        print("Sub-folder '{}' exists already".format(sub_folder))

folder_ctr = {}
i = 0

for file_name in files:
    i += 1
    file_path = os.path.join(dataset_path, file_name)
    if not is_valid_img(file_path):
        print("#{} '{}' skipping".format(i, file_name))
    else:
        sub_folder = get_split_assignment()
        print("#{} '{}' moving to {}".format(i, file_name, sub_folder))
        target_path = os.path.join(dataset_path, sub_folder, file_name)
        os.rename(file_path, target_path)

        if sub_folder not in folder_ctr:
            folder_ctr[sub_folder] = 0
        folder_ctr[sub_folder] += 1


print("Done")
exit(0)

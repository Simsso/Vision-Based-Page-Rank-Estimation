import logging
from glob import glob
import json
import os
from typing import Set, Union, Dict, Tuple, List
from torch.utils.data import Dataset
from rank_predictor.data import threefold
from rank_predictor.data.threefold import get_threefold
from rank_predictor.data.utils import Image, load_image, folder_to_rank, rank_to_logrank
from rank_predictor.data.v2.attributes import PageAttribute, LinkAttribute
from graph_nets import Attribute, Edge, Graph, Node


class DatasetV2(Dataset):
    """
    Dataset v2 of website graphs with associated rank.
    Each sample is a dictionary with
     * 'rank' being a tensor indicating the page rank
     * 'graph' being the graph representation of the page
     * 'logrank' being the logarithmically scaled rank (e.g. 80,001 and 90,000 are closer together than 1 and 10,000)
    """

    max_rank = 10 ** 5

    def __init__(self, page_paths: Set[str]) -> None:
        super(DatasetV2).__init__()

        self.page_paths = sorted(list(page_paths))

    def __getitem__(self, index) -> Dict[str, Union[int, Graph]]:
        page_path = self.page_paths[index]
        rank = folder_to_rank(page_path)

        sample = {
            'rank': rank,
            'logrank': rank_to_logrank(DatasetV2.max_rank, rank),
            'graph': self.load_graph(page_path)
        }
        return sample

    def __len__(self) -> int:
        return len(self.page_paths)

    @staticmethod
    def load_graph(path: str) -> Graph:
        # read JSON from page description file
        json_file_path = glob(os.path.join(path, '*.json'))
        assert len(json_file_path) == 1, "Number of json files in '{}' must be exactly one.".format(path)
        json_file_path = json_file_path[0]
        with open(json_file_path) as json_file:
            pages_json: List = json.load(json_file)

        # read screenshot paths
        imgs = DatasetV2.load_images(os.path.join(path, 'image'))

        assert len(pages_json) == len(imgs), "Number of pages and number of screenshots mismatch in '{}'.".format(path)

        # extract nodes
        nodes = {}
        for page_json in pages_json:
            desktop_img, mobile_img = imgs[page_json['id']-1]

            node_attribute = PageAttribute.from_json(page_json, desktop_img, mobile_img)

            url = page_json['base_url']
            if url in nodes:
                logging.debug("Found two nodes with the same URL.")
                continue
            node = Node(node_attribute)
            nodes[url] = node

        # extract edges
        edges = set()
        for page_json in pages_json:

            url = page_json['base_url']
            source_node = nodes[url]

            for edge_json in page_json.get('urls', []):

                target_url = edge_json['url']

                if target_url not in nodes:
                    logging.debug("Invalid link target URL. Could not find a node that corresponds to it.")
                    continue

                target_node = nodes[target_url]

                edge_attribute = LinkAttribute(target_url)
                edge = Edge(sender=source_node, receiver=target_node, attr=edge_attribute)

                edges.add(edge)

        return Graph(nodes=list(nodes.values()), edges=list(edges), attr=Attribute(None))

    @staticmethod
    def load_images(path: str) -> List[Tuple[Image, Image]]:
        file_extension = '.jpg'
        mobile_postfix = '_mobile'
        image_mobile_paths = set(glob(os.path.join(path, '*{}{}'.format(mobile_postfix, file_extension))))
        image_desktop_paths = set(glob(os.path.join(path, '*{}'.format(file_extension)))) - image_mobile_paths
        assert len(image_desktop_paths) == len(image_mobile_paths), \
            "Number of screenshots from mobile and desktop must be equal."

        images = []
        image_numbers = set()

        for desktop_p in image_desktop_paths:
            mobile_p = desktop_p[:-len(file_extension)] + mobile_postfix + desktop_p[-len(file_extension):]
            assert mobile_p in image_mobile_paths, "Found no mobile version for '{}' (desktop).".format(desktop_p)
            desktop_name = os.path.basename(desktop_p)[:-len(file_extension)]
            assert 0 < int(desktop_name) <= len(image_desktop_paths), "Found a desktop image with an invalid name."

            img_no = int(desktop_name)

            assert img_no not in image_numbers, "Found two images mapping to the same number."
            images.append((img_no, (load_image(desktop_p), load_image(mobile_p))))
            image_numbers.add(img_no)

        images.sort(key=lambda x: x[0])
        images = list(map(lambda x: x[1], images))  # remove keys

        return images

    @staticmethod
    def _get_page_paths(root_dir: str) -> Set[str]:
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        query_str = os.path.join(root_dir, '*', '')
        page_paths = sorted(glob(query_str))
        page_paths = set(page_paths)

        return set(page_paths)

    @staticmethod
    def from_path(root_dir: str):
        page_paths = DatasetV2._get_page_paths(root_dir)
        return DatasetV2(page_paths)

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
        assert os.path.isdir(root_dir), "The provided path '{}' is not a directory".format(root_dir)

        page_paths = list(DatasetV2._get_page_paths(root_dir))
        return get_threefold(DatasetV2, page_paths, train_ratio, valid_ratio)

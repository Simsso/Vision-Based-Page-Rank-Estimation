"""
Loads the dataset and feeds it through a feature extraction model.
It stores the model outputs and writes them into files.
"""

import logging
import os
from typing import Dict
import torch
from graph_nets.data_structures.attribute import Attribute
from torch import Tensor
from graph_nets.data_structures.graph import Graph
from torch.utils.data import DataLoader
from tqdm import tqdm
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from rank_predictor.model.graph_extractor_full import ScreenshotsFeatureExtractor

logging.basicConfig(level=logging.INFO)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
data = DatasetV2.from_path(dataset_dir, logrank_b=1)

# restore feature extractor
feat_extr_weights_path = os.path.expanduser(os.path.join(os.getenv('model_save_dir'), os.getenv('model_name')))
logging.info("Restoring pre-trained model weights")
net = ScreenshotsFeatureExtractor(drop_p=0)
net.load_state_dict(torch.load(feat_extr_weights_path)['extractor'])
net.eval()

if device.type == 'cuda':
    net.cuda()

# create data loader from dataset
data_loader: DataLoader = DataLoader(data, 1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

cache: Dict[int, Graph] = {}
file_ctr = 0


def save_cached_graphs(c: Dict[int, Graph], ctr: int) -> None:
    logging.info("Writing cached graphs to disk; #{}, len #{}".format(ctr, len(c)))
    torch.save(c, '{}-cache-{}.pt'.format(feat_extr_weights_path, ctr))


with torch.no_grad():
    for batch in tqdm(data_loader):
        g: Graph = batch[0]['graph'].to(device)
        g_hash = int(batch[0]['rank'])
        assert g_hash not in cache

        for n in g.nodes:

            desktop_img: Tensor = n.attr.val['desktop_img']
            mobile_img: Tensor = n.attr.val['mobile_img']

            # desktop and mobile feature vector
            x1, x2 = net(
                desktop_img.unsqueeze(0),
                mobile_img.unsqueeze(0)
            )

            x: torch.Tensor = torch.cat((x1, x2), dim=1).view(-1)
            x = x.detach().cpu()

            x = torch.Tensor(x.data)

            del n.attr
            n.attr = Attribute(x)

        g.to(torch.device('cpu'))

        cache[g_hash] = g

        if len(cache) == 4000:
            save_cached_graphs(cache, file_ctr)
            file_ctr += 1
            cache = {}

save_cached_graphs(cache, file_ctr)

# dataset
import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from rank_predictor.model.graph_extractor_full import ScreenshotsFeatureExtractor
from rank_predictor.model.graph_only_models import GNAvg

logging.basicConfig(level=logging.INFO)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
data = DatasetV2.from_path(dataset_dir, logrank_b=1)

# restore feature extractor
feat_extr_weights_path = os.path.expanduser(os.path.join(os.getenv('model_save_dir'), os.getenv('model_name')))
logging.info("Restoring pre-trained model weights")
feat_extr = ScreenshotsFeatureExtractor(drop_p=0)
feat_extr.load_state_dict(torch.load(feat_extr_weights_path))
net = GNAvg(feat_extr)

if device.type == 'cuda':
    feat_extr.cuda()
    net.cuda()

# create data loader from dataset
data_loader: DataLoader = DataLoader(data, 1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

cache = {}

for batch in tqdm(data_loader):
    with torch.no_grad():
        g = batch[0]['graph'].to(device)
        g_hash = batch[0]['rank']
        assert g_hash not in cache

        g_processed = net(g)

        cache[g_hash] = g_processed

torch.save(cache, '{}-cache.pt'.format(feat_extr_weights_path))

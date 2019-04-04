"""
Post-training (analysis) script searching for samples which excite individual units the most.
Secondly, we observe the latent variables, aka. feature maps of the conv layers.
This is relevant as part of the CNN understanding.

The method was introduced by Girshick et al. in their R-CNN paper, i.e. "Rich feature hierarchies for accurate object
detection and semantic segmentation" (http://arxiv.org/abs/1311.2524).
"""

from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
from graph_nets.data_structures.graph import Graph
from rank_predictor.analysis.utils import get_env_vars, restore_model, get_data
from rank_predictor.model.graph_extractor_full import GraphExtractorFull, ScreenshotsFeatureExtractor
import pickle


env = get_env_vars()
model: GraphExtractorFull = restore_model(env, GraphExtractorFull, {'num_core_blocks': 1, 'drop_p': 0.})
data = get_data(env)

cnn: ScreenshotsFeatureExtractor = model.screenshot_feature_extractor

a_desktop = []  # desktop activation vectors/maps
a_mobile = []  # mobile activation vectors/maps
r = []  # rank vector

iterator = tqdm(data)
for batch in iterator:
    sample = batch[0]
    g: Graph = sample['graph']
    this_r: int = sample['rank']

    node = g.ordered_nodes[0]

    desktop_img = node.attr.val['desktop_img']
    mobile_img = node.attr.val['mobile_img']
    img_no = node.attr.val['no']

    with torch.no_grad():
        d_feat, m_feat, d_act, m_act = cnn(
            desktop_img.unsqueeze(0),
            mobile_img.unsqueeze(0),
            return_feature_maps=True)

    # add outputs to activation dictionary
    d_act['feat'] = d_feat
    m_act['feat'] = m_feat

    # convert tensors into numpy arrays
    d_act: Dict[str, np.ndarray] = {k: d_act[k].view(d_act[k].shape[1:]).numpy() for k in d_act}
    m_act: Dict[str, np.ndarray] = {k: m_act[k].view(m_act[k].shape[1:]).numpy() for k in m_act}

    # add to accumulators
    r.append(this_r)
    a_desktop.append(d_act)
    a_mobile.append(m_act)

    if len(r) >= 100:
        iterator.close()
        break

result = {
    'rank': r,
    'feat_desktop': a_desktop,
    'feat_mobile': a_mobile
}

tqdm.write("Saving results")
with open('feature_maps.pickle', 'wb') as handle:
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

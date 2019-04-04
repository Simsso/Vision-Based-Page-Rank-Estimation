"""
Post-training analysis script that finds hard and easy samples.
"""

import torch
from tqdm import tqdm
from rank_predictor.analysis.utils import get_env_vars, restore_model, get_data
from rank_predictor.model.graph_extractor_full import GraphExtractorFull
import numpy as np
from rank_predictor.trainer.ranking.utils import per_sample_accuracy


env = get_env_vars()
model = restore_model(env, GraphExtractorFull, {'num_core_blocks': 1, 'drop_p': 0.})
data = get_data(env)

r = []  # vector of ranks
f = []  # model outputs vector

iterator = tqdm(data)
for batch in iterator:
    sample = batch[0]
    with torch.no_grad():
        model_out = model(sample['graph'])
    r.append(sample['rank'])
    f.append(model_out)

tqdm.write("Computing ranking and prediction matrices")
r: torch.Tensor = torch.Tensor(r)
f: torch.Tensor = torch.Tensor(f)

acc = per_sample_accuracy(r, f).numpy()
r: np.ndarray = r.numpy()
r = r.astype(np.int)

# vector of tuples (rank, accuracy of that sample)
rank_acc_vec = list(zip(r, acc))
rank_acc_vec_rank = sorted(rank_acc_vec, key=lambda t: t[0])
rank_acc_vec_acc = sorted(rank_acc_vec, key=lambda t: -t[1])

tqdm.write("Sorted by rank")
print(rank_acc_vec_rank)

tqdm.write("Sorted by accuracy")
print(rank_acc_vec_acc)

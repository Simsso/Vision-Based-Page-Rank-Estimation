"""
Post-training analysis script that finds hard and easy samples.
"""

import torch
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from tqdm import tqdm
from rank_predictor.analysis.utils import get_env_vars, restore_model, get_data, chunks, avg
import numpy as np
from rank_predictor.model.graph_only_models import GNAvg
from rank_predictor.trainer.ranking.utils import per_sample_accuracy


env = get_env_vars()
model = restore_model(env, GNAvg, {})
data = get_data(env, train_ratio=.6, valid_ratio=.2, cached=True)

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


batch_rank_span = 4000
max_rank = DatasetV2.max_rank+1
num_batches = max_rank//batch_rank_span
batch_r = [[] for _ in range(num_batches)]
batch_acc = [[] for _ in range(num_batches)]

for r, acc in rank_acc_vec_rank:
    batch = min(r // batch_rank_span, num_batches-1)
    batch_r[batch].append(r)
    batch_acc[batch].append(acc)

tqdm.write("Accuracy vs. rank")
for i in range(num_batches):
    r_mean = np.mean(batch_r[i])
    acc_mean = np.mean(batch_acc[i])
    stddev = np.std(batch_acc[i])
    tqdm.write("({}, {}) +- ({},{})".format(r_mean, acc_mean, stddev, stddev))

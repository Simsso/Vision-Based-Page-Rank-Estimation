"""
Post-training analysis script that finds hard and easy samples.
"""
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from rank_predictor.model.graph_extractor_full import GraphExtractorFull
import numpy as np

# env variables / parameters
from rank_predictor.trainer.ranking.probabilistic_loss import ProbabilisticLoss
from rank_predictor.trainer.ranking.utils import per_sample_accuracy

model_weights_dir = os.path.expanduser(os.getenv('weight_file_path'))
assert os.path.isfile(model_weights_dir), "Environment variable 'weight_file_path' must point to a file"

dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
assert os.path.isdir(dataset_dir), "Environment variables 'dataset_v2_path' must point to a directory"

# model
tqdm.write("Restoring model weights from '{}'".format(model_weights_dir))
device = torch.device('cpu')
model = GraphExtractorFull(num_core_blocks=1, drop_p=0.)
model.load_state_dict(torch.load(model_weights_dir, map_location=device))
model.eval()

# dataset
tqdm.write("Loading dataset")
train_ratio, valid_ratio = .85, .1
logrank_b = 1.5
data = DatasetV2.get_threefold(dataset_dir, train_ratio, valid_ratio, logrank_b)
data = DataLoader(data.test, 1, shuffle=True, num_workers=0, collate_fn=lambda b: b)

r = []  # vector of ranks
f = []  # model outputs vector

iterator = tqdm(data)
for batch in iterator:
    sample = batch[0]
    model_out = model(sample['graph'])
    r.append(sample['rank'])
    f.append(model_out)

    if len(r) >= 300:
        iterator.close()
        break

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

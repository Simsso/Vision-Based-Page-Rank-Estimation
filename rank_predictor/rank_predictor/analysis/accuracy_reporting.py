import os
from glob import glob
import torch
from tqdm import tqdm
from rank_predictor.trainer.ranking.utils import compute_batch_accuracy
from torch.utils.data import DataLoader, Dataset
from rank_predictor.data.v2.pagerank_dataset_cached import DatasetV2Cached
from rank_predictor.model.graph_only_models import GNDeep


def get_accuracy(dataset: Dataset) -> float:
    data_loader = DataLoader(dataset, 1, shuffle=False, num_workers=0, collate_fn=lambda b: b)
    with torch.no_grad():
        # accumulators
        model_outs, logranks = [], []

        for batch in data_loader:
            assert len(batch) == 1
            sample = batch[0]
            logrank: float = sample['logrank']
            graph: torch.Graph = sample['graph']
            model_out: torch.Tensor = model.forward(graph)

            model_outs.append(model_out)
            logranks.append(logrank)

        model_outs = torch.cat(model_outs)
        logranks = torch.Tensor(logranks)

        accuracy, _ = compute_batch_accuracy(target_ranks=logranks, model_outputs=model_outs)

        return accuracy


dataset_dir = os.path.expanduser(os.getenv('dataset_dir'))
feat_extr_weights_path = os.path.expanduser(os.getenv('feat_extr_weights_path'))
model_name = '10wob_deep_02'
edge_mode = 'default'
save_dir = os.path.expanduser('~/Development/pagerank/models')
save_files = glob(os.path.join(save_dir, '{}_0*.pt'.format(model_name)))
tqdm.write("Found #{} save files".format(len(save_files)))
print(save_files)

data = DatasetV2Cached.get_threefold(dataset_dir, train_ratio=0.6, valid_ratio=0.2, logrank_b=10,
                                     feat_extr_weights_path=feat_extr_weights_path)

valid_acc = []

model = GNDeep(drop_p=0, num_core_blocks=3, edge_mode=edge_mode, shared_weights=False)
model.eval()

tqdm.write("Model {}".format(model_name))

tqdm.write("Determining accuracy on validation set")

for save_file in tqdm(save_files):
    saved_weights = torch.load(save_file)
    model.load_state_dict(saved_weights)

    acc = get_accuracy(data.valid)
    valid_acc.append(acc)

max_valid_acc = max(valid_acc)
max_index = valid_acc.index(max_valid_acc)

print(valid_acc)
tqdm.write("Best at index {}, file {}".format(max_index, save_files[max_index]))

saved_weights = torch.load(save_files[max_index])
model.load_state_dict(saved_weights)

tqdm.write("Valid acc (max): {}".format(max_valid_acc))

test_acc = get_accuracy(data.test)
tqdm.write("Test acc: {}".format(test_acc))

train_acc = get_accuracy(data.train)
tqdm.write("Train acc: {}".format(train_acc))


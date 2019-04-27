import os
from typing import Dict, Callable, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rank_predictor.data.v2.pagerank_dataset import DatasetV2
from rank_predictor.data.v2.pagerank_dataset_cached import DatasetV2Cached


def get_env_vars() -> Dict[str, str]:
    """
    Reads environment variables, processes, and validates them.
    :return: Dictionary with the processed and validated environment variables.
    """
    model_weights_path = os.path.expanduser(os.getenv('weight_file_path'))
    assert os.path.isfile(model_weights_path), "Environment variable 'weight_file_path' must point to a file"

    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    assert os.path.isdir(dataset_dir), "Environment variables 'dataset_v2_path' must point to a directory"

    feat_extr_weights_path = os.path.expanduser(os.getenv('feat_extr_weights_path'))

    return {
        'model_weights_path': model_weights_path,
        'dataset_dir': dataset_dir,
        'feat_extr_weights_path': feat_extr_weights_path
    }


def restore_model(env: Dict[str, str], model_class: Callable, kwargs: Dict[str, any]) -> torch.nn.Module:
    # model
    tqdm.write("Restoring model weights from '{}'".format(env['model_weights_path']))
    device = torch.device('cpu')
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(env['model_weights_path'], map_location=device))
    model.eval()
    return model


def get_data(env: Dict[str, str], train_ratio: float = .85, valid_ratio: float = .1, cached: bool = False) -> DataLoader:
    tqdm.write("Loading dataset")
    if not cached:
        data = DatasetV2.get_threefold(env['dataset_dir'], train_ratio, valid_ratio, logrank_b=10)
    else:
        data = DatasetV2Cached.get_threefold(env['dataset_dir'], train_ratio, valid_ratio, logrank_b=10,
                                             feat_extr_weights_path=env['feat_extr_weights_path'])
    data = DataLoader(data.train, 1, shuffle=False, num_workers=0, collate_fn=lambda b: b)
    return data


def chunks(l: list, n: int) -> Iterable[list]:
    """yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def avg(l: list) -> float:
    assert len(l) > 0, "List must not be empty"
    return sum(l) / len(l)

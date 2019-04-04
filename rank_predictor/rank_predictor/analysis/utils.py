import os
from typing import Dict, Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from rank_predictor.data.v2.pagerank_dataset import DatasetV2


def get_env_vars() -> Dict[str, str]:
    """
    Reads environment variables, processes, and validates them.
    :return: Dictionary with the processed and validated environment variables.
    """
    model_weights_path = os.path.expanduser(os.getenv('weight_file_path'))
    assert os.path.isfile(model_weights_path), "Environment variable 'weight_file_path' must point to a file"

    dataset_dir = os.path.expanduser(os.getenv('dataset_v2_path'))
    assert os.path.isdir(dataset_dir), "Environment variables 'dataset_v2_path' must point to a directory"

    return {
        'model_weights_path': model_weights_path,
        'dataset_dir': dataset_dir
    }


def restore_model(env: Dict[str, str], model_class: Callable, kwargs: Dict[str, any]) -> torch.nn.Module:
    # model
    tqdm.write("Restoring model weights from '{}'".format(env['model_weights_path']))
    device = torch.device('cpu')
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(env['model_weights_path'], map_location=device))
    model.eval()
    return model


def get_data(env: Dict[str, str], train_ratio: float = .85, valid_ratio: float = .1) -> DataLoader:
    tqdm.write("Loading dataset")
    data = DatasetV2.get_threefold(env['dataset_dir'], train_ratio, valid_ratio, logrank_b=2)
    data = DataLoader(data.test, 1, shuffle=False, num_workers=0, collate_fn=lambda b: b)
    return data

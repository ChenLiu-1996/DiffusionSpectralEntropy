import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath('../'))
from barlowtwins_model import BarlowTwinsModel
from moco_model import MoCoModel
from simsiam_model import SimSiamModel
from swav_model import SwavModel


def run_model(
        model_name: str,
        version: str = None,
        device: torch.device = None,
        random_seed: int = 1) -> None:

    base_version = {
        'barlowtwins': 'barlowtwins_bs2048_ep1000',
        'moco': 'moco_v1_ep200',
        'simsiam': 'simsiam_bs256_ep100',
        'swav': 'swav_bs256_ep200',
    }
    if version is None:
        version = base_version[model_name]

    if model_name == 'barlowtwins':
        model = BarlowTwinsModel(device=device, version=version)
    elif model_name == 'moco':
        model = MoCoModel(device=device, version=version)
    elif model_name == 'simsiam':
        model = SimSiamModel(device=device, version=version)
    elif model_name == 'swav':
        model = SwavModel(device=device, version=version)
    else:
        raise ValueError('model_name: %s not supported.' % model_name)

    model.restore_model()
    model.eval()

    noise = torch.from_numpy(
        np.random.RandomState(random_seed).randn(
            1, 3, 256, 256)).float().to(device)

    with torch.no_grad():
        _ = model.forward(noise)

    print('\n\nFetched Latent Features:')
    latent_outputs = model.fetch_latent()
    for key in latent_outputs.keys():
        print(key, latent_outputs[key].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--version', type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_model(model_name=args.model, version=args.version, device=device)

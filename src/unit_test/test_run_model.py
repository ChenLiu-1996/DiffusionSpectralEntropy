import os
import sys

import numpy as np
import torch

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/nn/external_model_loader/')
from barlowtwins_model import BarlowTwinsModel
from moco_model import MoCoModel
from simsiam_model import SimSiamModel
from swav_model import SwavModel
from vicreg_model import VICRegModel
from vicregl_model import VICRegLModel


def exclude_bias_and_norm(p):
    '''
    This function being here is important.
    Without it, the model VICRegL won't properly load.
    '''
    return p.ndim == 1


def run_model(
        device: torch.device = None,
        random_seed: int = 1) -> None:

    base_version = {
        # 'barlowtwins': 'barlowtwins_bs2048_ep1000',
        # 'moco': 'moco_v1_ep200',
        # 'simsiam': 'simsiam_bs256_ep100',
        # 'swav': 'swav_bs256_ep200',
        # 'vicreg': 'vicreg_bs2048_ep100',
        'vicregl': 'vicregl_alpha0d75_bs2048_ep300',
    }

    for model_name in base_version.keys():
        version = base_version[model_name]

        print('Testing model: %s' % model_name)

        if model_name == 'barlowtwins':
            model = BarlowTwinsModel(device=device, version=version)
        elif model_name == 'moco':
            model = MoCoModel(device=device, version=version)
        elif model_name == 'simsiam':
            model = SimSiamModel(device=device, version=version)
        elif model_name == 'swav':
            model = SwavModel(device=device, version=version)
        elif model_name == 'vicreg':
            model = VICRegModel(device=device, version=version)
        elif model_name == 'vicregl':
            model = VICRegLModel(device=device, version=version)
        else:
            raise ValueError('model_name: %s not supported.' % model_name)

        model.restore_model()
        model.eval()

        noise = torch.from_numpy(
            np.random.RandomState(random_seed).randn(
                1, 3, 256, 256)).float().to(device)

        with torch.no_grad():
            _ = model.forward(noise)

        print('Success')


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_model(device=device)

import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('./'))
from moco_model import MoCoModel


def extract_features(model_name: str,
                     load_weights: bool = True,
                     noise_list: List[torch.Tensor] = None,
                     device: torch.device = None):

    __models = ['barlowtwins', 'moco', 'simsiam', 'swav']
    assert model_name in __models

    __versions = {
        'moco': ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'],
    }
    top1_acc = {
        'moco': [60.6, 67.7, 71.1],
    }
    summary = {}

    for i, version in enumerate(__versions[model_name]):
        summary[version] = {
            'manifold_features': None,
            'top1_acc': top1_acc[model_name][i],
        }

        if model_name == 'moco':
            model = MoCoModel(device=device, version=version)

        if load_weights:
            model.restore_model()
        model.eval()

        embeddings = []
        with torch.no_grad():
            for noise in tqdm(noise_list):
                _ = model.forward(noise)
                embedding_dict = model.fetch_latent()
                embeddings.append(
                    embedding_dict['avgpool'].cpu().detach().numpy().squeeze(0))
        embeddings = np.array(embeddings)

        embeddings = embeddings.squeeze()

        VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
        point_clouds = embeddings[None, ...]
        diagrams = VR.fit_transform(point_clouds)
        PE = PersistenceEntropy()
        features = PE.fit_transform(diagrams)

        summary[version]['manifold_features'] = features

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_resolution = (3, 256, 256)  # Channel first.

    noises = torch.from_numpy(
        np.random.RandomState(args.random_seed).randn(
            args.num_points, *image_resolution)).float().to(device)

    # Cannot fit a tensor with a big batch size through the model.
    # Convert to a list of tensors with B = 1.
    noise_list = [noises[i, ...][None, ...] for i in range(noises.shape[0])]

    summary = extract_features(
        model_name='moco', noise_list=noise_list, device=device)

    import pdb
    pdb.set_trace()

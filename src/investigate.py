import argparse
from typing import List

import numpy as np
import ripserplusplus as rpp_py
import torch
from barlowtwins_model import BarlowTwinsModel
from gtda.diagrams import BettiCurve, PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from matplotlib import pyplot as plt
from moco_model import MoCoModel
from simsiam_model import SimSiamModel
from swav_model import SwavModel
from tqdm import tqdm


def ripser2gtda(dgm, max_dim):
    diags = []
    for dim in range(max_dim + 1):
        num_pts = len(dgm[dim])
        pers_diag = np.zeros((num_pts, 3))
        for idx in range(num_pts):
            pers_diag[idx, 0] = dgm[dim][idx][0]
            pers_diag[idx, 1] = dgm[dim][idx][1]
            pers_diag[idx, 2] = dim
        diags.append(pers_diag)

    return np.vstack(diags)


def extract_features(load_weights: bool = True,
                     noise_list: List[torch.Tensor] = None,
                     device: torch.device = None):

    __models = ['barlowtwins', 'moco', 'simsiam', 'swav']
    __versions = {
        'barlowtwins': ['barlowtwins_bs2048_ep1000'],
        'moco': ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'],
        'simsiam': ['simsiam_bs256_ep100', 'simsiam_bs512_ep100'],
        'swav': [
            'swav_bs256_ep200', 'swav_bs256_ep400',
            'swav_bs4096_ep100', 'swav_bs4096_ep200',
            'swav_bs4096_ep400', 'swav_bs4096_ep800',
        ],
    }
    top1_acc = {
        'barlowtwins': [73.5],
        'moco': [60.6, 67.7, 71.1],
        'simsiam': [68.3, 68.1],
        'swav': [72.7, 74.3, 72.1, 73.9, 74.6, 75.3],
    }
    summary = {}

    for model_name in __models:
        for i, version in enumerate(__versions[model_name]):
            summary[version] = {
                'manifold_features': None,
                'top1_acc': top1_acc[model_name][i],
            }

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
            # point_clouds = embeddings[None, ...]
            print(embeddings.shape)
            # diagrams = VR.fit_transform(point_clouds)
            # PE = PersistenceEntropy()
            # features = PE.fit_transform(diagrams)

            # BT = BettiCurve()
            # betti = BT.fit_transform(diagrams)
            max_dim = 2
            vrp = rpp_py.run("--dim %s --format point-cloud" %
                             max_dim, embeddings)
            diagram = ripser2gtda(vrp, max_dim=max_dim)

            # TODO:
            # Implement the following
            # 1. Real ImageNet data as input.
            # 2. How to calculate (high) curvature towards class center and (low) curvature outside?
            # 3. How to calculate (high) density/volume towards class center and (low) density/volume outside?
            # 4. How to extract the [0, 1, 2, 3] homology features of manifolds?

            import pdb
            pdb.set_trace()

            # summary[version]['manifold_features'] = features

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

    summary = extract_features(noise_list=noise_list, device=device)

    points = []
    colors = []
    candidates = []
    for candidate in summary.keys():
        candidates.append(candidate)
        feature_acc_dict = summary[candidate]
        points.append(feature_acc_dict['manifold_features'])
        colors.append(feature_acc_dict['top1_acc'])
    points = np.array(points)
    points = points.squeeze(1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1],
                    points[:, 2], c=colors, alpha=0.8)
    plt.colorbar(sc)
    fig.savefig('manifold_features.png')

import argparse
import os
import sys
from typing import List, Union

import numpy as np
import seaborn as sns
import torch
import torchvision
from diffusion_curvature.core import DiffusionMatrix
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from seed import seed_everything

sys.path.insert(0, import_dir + '/external_model_loader/')
from barlowtwins_model import BarlowTwinsModel
from moco_model import MoCoModel
from simsiam_model import SimSiamModel
from swav_model import SwavModel
from vicreg_model import VICRegModel


def von_neumann_entropy(eigs, trivial_thr: float = 0.9):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the biggest eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues <= trivial_thr]

    # Shift the negative eigenvalue(s).
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))


def compute_diffusion_entropy(embeddings: torch.Tensor, args: AttributeHashmap,
                              vne_thr_list: List[float], eig_npy_path: str):

    if os.path.exists(eig_npy_path):
        data_numpy = np.load(eig_npy_path)
        eigenvalues_P = data_numpy['eigenvalues_P']
        print('Pre-computed eigenvalues loaded.')
    else:
        # Diffusion Matrix
        diffusion_matrix = DiffusionMatrix(embeddings,
                                           kernel_type="adaptive anisotropic",
                                           k=args.knn)
        print('Diffusion matrix computed.')

        # Diffusion Eigenvalues
        eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
        print('Eigenvalues computed.')

        with open(eig_npy_path, 'wb+') as f:
            np.savez(f, eigenvalues_P=eigenvalues_P)

    # von Neumann Entropy
    vne_list = []
    for trivial_thr in vne_thr_list:
        vne = von_neumann_entropy(eigenvalues_P, trivial_thr=trivial_thr)
        vne_list.append(vne)

    return vne_list


def get_val_loader(args: AttributeHashmap) -> torch.utils.data.DataLoader:
    dataset_dir = '%s/data/%s' % (args.root_dir, args.dataset)

    if args.dataset == 'mnist':
        args.in_channels = 1
        args.num_classes = 10
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset_loader = torchvision.datasets.MNIST

    elif args.dataset == 'cifar10':
        args.in_channels = 3
        args.num_classes = 10
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset_loader = torchvision.datasets.CIFAR10

    elif args.dataset == 'cifar100':
        args.in_channels = 3
        args.num_classes = 100
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset_loader = torchvision.datasets.CIFAR100

    elif args.dataset == 'stl10':
        args.in_channels = 3
        args.num_classes = 10
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset_loader = torchvision.datasets.STL10

    else:
        raise ValueError(
            '`args.dataset` value not supported. Value provided: %s.' %
            args.dataset)

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        val_loader = torch.utils.data.DataLoader(torchvision_dataset_loader(
            dataset_dir, train=False, download=False, transform=transform_val),
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 shuffle=False)

    elif args.dataset in ['stl10']:
        val_loader = torch.utils.data.DataLoader(
            torchvision_dataset_loader(dataset_dir,
                                       split='test',
                                       download=False,
                                       transform=transform_val),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False)

    return val_loader


def diffusion_entropy(load_weights: bool = True,
                      dataloader: torch.utils.data.DataLoader = None,
                      device: torch.device = None,
                      npy_folder: str = None):

    __models = ['barlowtwins', 'moco', 'simsiam', 'swav', 'vicreg']
    __versions = {
        'barlowtwins': ['barlowtwins_bs2048_ep1000'],
        'moco': ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'],
        'simsiam': ['simsiam_bs256_ep100', 'simsiam_bs512_ep100'],
        'swav': [
            'swav_bs256_ep200',
            'swav_bs256_ep400',
            'swav_bs4096_ep100',
            'swav_bs4096_ep200',
            'swav_bs4096_ep400',
            'swav_bs4096_ep800',
        ],
        'vicreg': ['vicreg_bs2048_ep100']
    }
    top1_acc = {
        'barlowtwins': [73.5],
        'moco': [60.6, 67.7, 71.1],
        'simsiam': [68.3, 68.1],
        'swav': [72.7, 74.3, 72.1, 73.9, 74.6, 75.3],
        'vicreg': [73.2],
    }
    summary = {'vne_thr_list': [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.00]}

    for model_name in __models:
        for i, version in enumerate(__versions[model_name]):
            print('model: %s, version: %s' % (model_name, version))

            summary[version] = {
                'top1_acc': top1_acc[model_name][i],
                'vne_list': [],
            }

            embedding_npy_path = '%s/%s_embeddings.npy' % (npy_folder, version)
            eig_npy_path = '%s/%s_eigP.npy' % (npy_folder, version)

            if os.path.exists(embedding_npy_path):
                data_numpy = np.load(embedding_npy_path)
                embeddings = data_numpy['embeddings']
                print('Pre-computed embeddings loaded.')
            else:
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
                else:
                    raise ValueError('model_name: %s not supported.' %
                                     model_name)

                if load_weights:
                    model.restore_model()
                model.eval()

                embeddings = []
                with torch.no_grad():
                    for i, (x, y_true) in enumerate(tqdm(dataloader)):
                        assert args.in_channels in [1, 3]
                        if args.in_channels == 1:
                            # Repeat the channel dimension: 1 channel -> 3 channels.
                            x = x.repeat(1, 3, 1, 1)
                        x, y_true = x.to(device), y_true.to(device)

                        _ = model.forward(x)
                        embedding_dict = model.fetch_latent()
                        # shape: [B, d, 1, 1]
                        embedding_vec = embedding_dict['avgpool'].cpu().detach(
                        ).numpy()
                        # shape: [B, d]
                        embedding_vec = embedding_vec.reshape(
                            embedding_vec.shape[:2])
                        embeddings.append(embedding_vec)

                embeddings = np.concatenate(embeddings)
                print('Embeddings computed.')

                with open(embedding_npy_path, 'wb+') as f:
                    np.savez(f, embeddings=embeddings)

            summary[version]['vne_list'] = compute_diffusion_entropy(
                embeddings,
                args,
                summary['vne_thr_list'],
                eig_npy_path=eig_npy_path)

    return summary


def normalize(
        data: Union[np.array, torch.Tensor],
        dynamic_range: List[float] = [0, 1]) -> Union[np.array, torch.Tensor]:
    assert len(dynamic_range) == 2

    x1, x2 = data.min(), data.max()
    y1, y2 = dynamic_range[0], dynamic_range[1]

    slope = (y2 - y1) / (x2 - x1)
    offset = (y1 * x2 - y2 * x1) / (x2 - x1)

    data = data * slope + offset

    # Fix precision issue.
    data = data.clip(y1, y2)
    return data


def plot_summary(summary: dict, fig_prefix: str = None):
    version_list, vne_list, acc_list = [], [], []
    vne_thr_list = summary['vne_thr_list']

    fig_vne = plt.figure(figsize=(10, 8))
    fig_corr = plt.figure(figsize=(8, 6 * len(vne_thr_list)))

    ax = fig_vne.add_subplot(1, 1, 1)
    for key in summary.keys():
        if key == 'vne_thr_list':
            continue
        version = key
        version_list.append(version)
        vne_list.append(summary[version]['vne_list'])
        acc_list.append(summary[version]['top1_acc'])

    vne_array = np.array(vne_list)
    acc_list = np.array(acc_list)
    scaled_acc = normalize(acc_list,
                           dynamic_range=[np.min(vne_list),
                                          np.max(vne_list)])

    ax.plot(version_list, vne_array)
    ax.legend(vne_thr_list)
    ax.scatter(version_list, scaled_acc)
    ax.set_xticks(np.arange(len(version_list)))
    ax.set_xticklabels(version_list, rotation=90)
    ax.spines[['right', 'top']].set_visible(False)

    fig_vne.tight_layout()
    fig_vne.savefig(fig_prefix + '-vne.png')

    for thr_idx in range(len(vne_thr_list)):
        ax = fig_corr.add_subplot(len(vne_thr_list), 1, thr_idx + 1)
        ax.scatter(acc_list, vne_array[..., thr_idx])
        pearson_r, pearson_p = pearsonr(acc_list, vne_array[..., thr_idx])
        spearman_r, spearman_p = spearmanr(acc_list, vne_array[..., thr_idx])
        ax.set_title(
            '[Diffusion Entropy] removing eigenvalues > %s \nP.R: %.3f (p = %.3f), S.R: %.3f (p = %.3f)'
            % (vne_thr_list[thr_idx], pearson_r, pearson_p, spearman_r,
               spearman_p))
        ax.set_xticks(acc_list)
        ax.set_xticklabels(version_list, rotation=90)
        ax.spines[['right', 'top']].set_visible(False)

    fig_corr.tight_layout()
    fig_corr.savefig(fig_prefix + '-correlation.png')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id',
                        help='which gpu to use',
                        type=int,
                        default=0)
    parser.add_argument('--dataset',
                        help='which dataset to run',
                        type=str,
                        default='mnist')
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument('--seed', help='random seed.', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=1)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    seed_everything(args.seed)

    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    args.root_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])

    save_folder = './results_diffusion_entropy_PublicModels/%s' % args.dataset
    npy_folder = '%s/%s/' % (save_folder, 'numpy_files')
    os.makedirs(npy_folder, exist_ok=True)

    val_loader = get_val_loader(args=args)
    summary = diffusion_entropy(dataloader=val_loader,
                                device=device,
                                npy_folder=npy_folder)
    fig_prefix = '%s/diffusion-entropy-PublicModels-%s' % (save_folder,
                                                           args.dataset)
    plot_summary(summary, fig_prefix=fig_prefix)

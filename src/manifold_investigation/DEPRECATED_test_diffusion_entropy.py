import argparse
import os
import sys
from glob import glob

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from typing import Dict, Iterable
import random
import seaborn as sns

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from information import von_neumann_entropy
from diffusion import compute_diffusion_matrix
from path_utils import update_config_dirs
from seed import seed_everything

cifar10_int2name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


def shift_entropy(eigs: np.array, noise_eigval_thr: float = 1e-3):
    eigenvalues = eigs.copy()
    eigenvalues = eigenvalues.astype(np.float64)  # mitigates rounding error.

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
    eigenvalues = eigenvalues[1:]

    # Eigenvalues may be negative. Only care about the magnitude, not the sign.
    eigenvalues -= eigenvalues.min()

    # Drop the trivial eigenvalues that are corresponding to noise eigenvectors.
    eigenvalues[eigenvalues < noise_eigval_thr] = 0

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log2(prob))

def plot_figures(data_arrays: Dict[str, Iterable],
                 save_paths_fig: Dict[str, str]) -> None:

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    # Plot of Diffusion Entropy vs. epoch.
    fig_vne = plt.figure(figsize=(20, 10))
    ax = fig_vne.add_subplot(1, 2, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['acc'], data_arrays['vne'], c='mediumblue', s=120)
    ax.plot(data_arrays['acc'], data_arrays['vne'], c='mediumblue')
    fig_vne.supylabel('Diffusion Entropy', fontsize=40)
    fig_vne.supxlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    fig_vne.savefig(save_paths_fig['fig_vne'])
    plt.close(fig=fig_vne)

    ax = fig_vne.add_subplot(1, 2, 2)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['acc'],
               data_arrays['vne_topk'],
               c='mediumblue',
               s=120)
    ax.plot(data_arrays['acc'], data_arrays['vne_topk'], c='mediumblue')
    fig_vne.supylabel('Diffusion Entropy', fontsize=40)
    fig_vne.supxlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    fig_vne.savefig(save_paths_fig['fig_vne'])
    plt.close(fig=fig_vne)

    fig_eigval = plt.figure(figsize=(6, 2 * len(data_arrays['eigval'])))
    for i, eigvals in enumerate(data_arrays['eigval']):
        ax = fig_eigval.add_subplot(len(data_arrays['eigval']), 1, i + 1)
        sns.boxplot(x=eigvals, color='skyblue', ax=ax)
    plt.tight_layout()
    fig_eigval.savefig(save_paths_fig['fig_eigval'])
    plt.close(fig=fig_eigval)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument(
        '--random-seed',
        help='Only enter if you want to override the config!!!',
        type=int,
        default=None)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))
    if args.random_seed is not None:
        config.random_seed = args.random_seed

    seed_everything(config.random_seed)

    if 'contrastive' in config.keys():
        method_str = config.contrastive
    elif 'bad_method' in config.keys():
        method_str = config.bad_method

    # embedding_folders = sorted(
    #     glob('%s/embeddings/%s-%s-%s-seed%s-val_acc*' %
    #          (config.output_save_path, config.dataset, method_str,
    #           config.model, config.random_seed)))
    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-%s-seed%s-epoch*' %
             (config.output_save_path, config.dataset, method_str,
              config.model, config.random_seed)))

    save_root = './results_test/'
    os.makedirs(save_root, exist_ok=True)

    save_paths_fig = {
        'fig_vne':
        '%s/diffusion-entropy-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn),
        'fig_eigval':
        '%s/eigvals-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn),
    }

    acc_list, vne_list, vne_topk_list, eigval_list = [], [], [], []

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))
        checkpoint_name = os.path.basename(embedding_folder)

        # acc_list.append(checkpoint_name.split('_')[-1])
        # if '%' in acc_list[-1]:
        #     acc_list.append(int(acc_list[-1].split('%')[0]) / 100)
        # else:
        #     acc_list.append(acc_list[-1] + 0.1)
        acc_list.append(
            float(
                embedding_folder.split('-valAcc')[1].split('-divergence')[0]))

        labels, embeddings, orig_input = None, None, None

        for file in tqdm(files):
            np_file = np.load(file)
            curr_input = np_file['image']
            curr_label = np_file['label_true']
            curr_embedding = np_file['embedding']

            if labels is None:
                orig_input = curr_input
                labels = curr_label[:, None]  # expand dim to [B, 1]
                embeddings = curr_embedding
            else:
                orig_input = np.vstack((orig_input, curr_input))
                labels = np.vstack((labels, curr_label[:, None]))
                embeddings = np.vstack((embeddings, curr_embedding))

        # This is the matrix of N embedding vectors each at dim [1, D].
        N, D = embeddings.shape

        assert labels.shape[0] == N
        assert labels.shape[1] == 1

        if config.dataset == 'cifar10':
            labels_updated = np.zeros(labels.shape, dtype='object')
            for k in range(N):
                labels_updated[k] = cifar10_int2name[labels[k].item()]
            labels = labels_updated
            del labels_updated

        #
        '''Diffusion Matrix and Diffusion Eigenvalues'''
        save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-%s.npz' % (
            'results_diffusion_entropy', checkpoint_name)
        os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
        if os.path.exists(save_path_eigenvalues):
            data_numpy = np.load(save_path_eigenvalues)
            eigenvalues_P = data_numpy['eigenvalues_P']
            print('Pre-computed eigenvalues loaded.')
        else:
            break

        #
        '''Diffusion Entropy'''
        vne = von_neumann_entropy(eigenvalues_P)
        vne_list.append(vne)

        # eigenvalues_P_topk = eigenvalues_P.copy()
        # eigenvalues_P_topk = np.array(sorted(eigenvalues_P_topk)[::-1][:100])
        eigenvalues_shift = eigenvalues_P.copy()
        vne_topk = shift_entropy(eigenvalues_shift)
        # vne_topk = von_neumann_entropy(eigenvalues_P_topk)
        vne_topk_list.append(vne_topk)

        #
        '''Eigenvalues'''
        eigval_list.append(eigenvalues_P)

        # Plotting
        data_arrays = {
            'acc': acc_list,
            'vne': vne_list,
            'vne_topk': vne_topk_list,
            'eigval': eigval_list,
        }
        plot_figures(data_arrays=data_arrays, save_paths_fig=save_paths_fig)

import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn import datasets
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from information import exact_eigvals, von_neumann_entropy, shannon_entropy
from diffusion import compute_diffusion_matrix

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy
from dsmi import diffusion_spectral_mutual_information


def force_aspect(ax):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    aspect = (xmax_ - xmin_) / (ymax_ - ymin_)
    ax.set_aspect(aspect=aspect, adjustable='box')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', action='store_true')

    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_classes = 5
    cluster_std = 1
    cluster_std_lists = [1.0, 1.0, 1.0, 1.0, 1.0]
    center_box = (-30.0, 30.0)

    N = 1000 * num_classes
    save_path_fig = '%s/intuitive_DSE.png' % (save_root)

    D_list = [2, 3]
    t_list = list(range(5))

    separated_DSEs = [[[] for _ in range(len(t_list))]
                       for _ in range(len(D_list))]
    single_DSEs = [[[] for _ in range(len(t_list))]
                    for _ in range(len(D_list))]

    # Multiple blobs
    for d, dim in enumerate(D_list):
        embeddings, labels = datasets.make_blobs(
            n_samples=N,
            n_features=dim,
            centers=num_classes,
            cluster_std=cluster_std_lists,
            center_box=center_box)
        labels = labels.reshape(N, 1)
        if args.transform == True:
            transformation = np.random.normal(loc=0,
                                                scale=0.3,
                                                size=(dim, dim))  # D x D
            embeddings = np.dot(embeddings, transformation)
        embeddings /= np.sqrt(dim)
        
        # DSE
        for j, t in tqdm(enumerate(t_list)):
            diffusion_matrix = compute_diffusion_matrix(embeddings, sigma=np.sqrt(dim))
            eigvals = exact_eigvals(diffusion_matrix)
            eigvals = np.abs(eigvals)
            # Power eigenvalues to `t` to mitigate effect of noise.
            eigvals = eigvals**t
            prob = eigvals / eigvals.sum()
            prob = prob + np.finfo(float).eps
            dse = -np.sum(prob * np.log2(prob))

            separated_DSEs[d][j] = dse


    # Single blob
    for d, dim in enumerate(D_list):
        embeddings, _ = datasets.make_blobs(n_samples=N,
                                            n_features=dim,
                                            centers=1,
                                            cluster_std=cluster_std)
        embeddings /= np.sqrt(dim)
        num_per_class = N // num_classes
        labels = np.zeros(N)
        # Randomly assign class labels
        for i in range(1, num_classes + 1):
            inds = np.random.choice(np.where(labels == 0)[0],
                                    num_per_class,
                                    replace=False)
            labels[inds] = i
        labels = labels - 1  # class 1,2,3 -> class 0,1,2
        labels = labels.reshape(N, 1)

        #DSE
        for j, t in tqdm(enumerate(t_list)):
            diffusion_matrix = compute_diffusion_matrix(embeddings, sigma=np.sqrt(dim))
            eigvals = exact_eigvals(diffusion_matrix)
            eigvals = np.abs(eigvals)
            # Power eigenvalues to `t` to mitigate effect of noise.
            eigvals = eigvals**t
            prob = eigvals / eigvals.sum()
            prob = prob + np.finfo(float).eps
            dse = -np.sum(prob * np.log2(prob))

            single_DSEs[d][j] = dse

    separated_DSEs = np.array(separated_DSEs)
    single_mi_DSEs = np.array(single_DSEs)

    print(separated_DSEs.shape, single_mi_DSEs)

    fig_mi = plt.figure(figsize=(28, 10))
    gs = GridSpec(3, 4, figure=fig_mi)

    for dim, gs_x, gs_y in zip([2, 3], [0, 0], [0, 1]):
        if dim == 3:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y], projection='3d')
        else:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y])
        ax.spines[['right', 'top']].set_visible(False)

        embeddings, labels = datasets.make_blobs(n_samples=N,
                                                 n_features=dim,
                                                 centers=num_classes,
                                                 cluster_std=cluster_std_lists,
                                                 center_box=center_box,
                                                 random_state=0)
        if args.transform == True:
            #transformation = np.random.randn(dim, dim)# D x D
            transformation = np.random.normal(loc=0,
                                              scale=0.3,
                                              size=(dim, dim))  # D x D
            embeddings = np.dot(embeddings, transformation)
        if dim == 2:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
                force_aspect(ax)
        elif dim == 3:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           embeddings[inds, 2],
                           color=color,
                           alpha=0.5)


    for dim, gs_x, gs_y in zip([2, 3], [0, 0], [2, 3]):
        if dim == 3:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y], projection='3d')
        else:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y])
        ax.spines[['right', 'top']].set_visible(False)

        embeddings, _ = datasets.make_blobs(n_samples=N,
                                            n_features=dim,
                                            centers=1,
                                            cluster_std=cluster_std,
                                            random_state=0)
        num_per_class = N // num_classes
        labels = np.zeros(N)
        # Randomly assign class labels
        for i in range(1, num_classes + 1):
            inds = np.random.choice(np.where(labels == 0)[0],
                                    num_per_class,
                                    replace=False)
            labels[inds] = i
        labels = labels - 1  # class 1,2,3 -> class 0,1,2

        if dim == 2:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
                force_aspect(ax)
        elif dim == 3:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           embeddings[inds, 2],
                           color=color,
                           alpha=0.5)

    # DSE vs. t
    ymin = min([np.min(separated_DSEs[0:2]),
                np.min(single_DSEs[0:2])]) - 0.1
    ymax = max([np.max(separated_DSEs[0:2]),
                np.max(single_DSEs[0:2])]) + 0.1
    
    ax = fig_mi.add_subplot(gs[1:3, 0:1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(separated_DSEs[0])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.set_ylabel('DSE', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_mi.add_subplot(gs[1:3, 1:2])
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(separated_DSEs[1])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)


    ax = fig_mi.add_subplot(gs[1:3, 2:3])
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(single_DSEs[0])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_mi.add_subplot(gs[1:3, 3:4])
    ax.spines[['right', 'top']].set_visible(False)
    ax.plot(single_DSEs[1])
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig)
    plt.close(fig=fig_mi)
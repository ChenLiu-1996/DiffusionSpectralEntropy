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
from information import mutual_information_per_class_random_sample


def force_aspect(ax):
    xmin_, xmax_ = ax.get_xlim()
    ymin_, ymax_ = ax.get_ylim()
    aspect = (xmax_ - xmin_) / (ymax_ - ymin_)
    ax.set_aspect(aspect=aspect, adjustable='box')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=1)
    parser.add_argument('--transform', action='store_true')

    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_classes = 3
    cluster_std = 1
    cluster_std_lists = [1.0, 1.0, 1.0]
    center_box = (-30.0, 30.0)

    N = 1000 * num_classes
    save_path_fig = '%s/intuitive-DSMI.png' % (save_root)

    num_repetition = 5

    D_list = [2, 3]
    t_list = [1, 2, 3, 4, 5, 10, 100, 1000]

    separated_mi_Ys = [[[[] for _ in range(num_repetition)]
                        for _ in range(len(t_list))]
                       for _ in range(len(D_list))]
    single_mi_Ys = [[[[] for _ in range(num_repetition)]
                     for _ in range(len(t_list))] for _ in range(len(D_list))]

    for k in tqdm(range(num_repetition)):
        for d, dim in enumerate(D_list):
            embeddings, labels = datasets.make_blobs(
                n_samples=N,
                n_features=dim,
                centers=num_classes,
                cluster_std=cluster_std_lists,
                center_box=center_box,
                random_state=k)
            labels = labels.reshape(N, 1)
            if args.transform == True:
                transformation = np.random.normal(loc=0,
                                                  scale=0.3,
                                                  size=(dim, dim))  # D x D
                embeddings = np.dot(embeddings, transformation)
            embeddings /= np.sqrt(dim)
            embeddings[:, 0] = embeddings[:, 0] * 3
            for j, t in enumerate(t_list):
                mi_Y, _, H_ZgivenY = mutual_information_per_class_random_sample(
                    embeddings=embeddings,
                    labels=labels,
                    H_ZgivenY_map=None,
                    sigma=args.gaussian_kernel_sigma,
                    vne_t=t,
                )
                separated_mi_Ys[d][j][k] = mi_Y

    for k in tqdm(range(num_repetition)):
        for d, dim in enumerate(D_list):
            embeddings, _ = datasets.make_blobs(n_samples=N,
                                                n_features=dim,
                                                centers=1,
                                                cluster_std=cluster_std,
                                                random_state=k)
            embeddings /= np.sqrt(dim)
            embeddings[:, 0] = embeddings[:, 0] * 3
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

            for j, t in enumerate(t_list):
                mi_Y, _, _ = mutual_information_per_class_random_sample(
                    embeddings=embeddings,
                    labels=labels,
                    H_ZgivenY_map=None,
                    sigma=args.gaussian_kernel_sigma,
                    vne_t=t,
                )
                single_mi_Ys[d][j][k] = mi_Y

    separated_mi_Ys = np.array(separated_mi_Ys)
    single_mi_Ys = np.array(single_mi_Ys)
    # print(separated_mi_Ys.shape, single_mi_Ys)

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
        embeddings[:, 0] = embeddings[:, 0] * 3
        if dim == 2:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
                # force_aspect(ax)
        elif dim == 3:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
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
        embeddings[:, 0] = embeddings[:, 0] * 3
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
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
                # force_aspect(ax)
        elif dim == 3:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           embeddings[inds, 2],
                           color=color,
                           alpha=0.5)

    # MI vs. t
    ymin = min([np.min(separated_mi_Ys[0:2].T),
                np.min(single_mi_Ys[0:2].T)]) - 0.1
    ymax = max([np.max(separated_mi_Ys[0:2].T),
                np.max(single_mi_Ys[0:2].T)]) + 0.1
    ax = fig_mi.add_subplot(gs[1:3, 0:1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.boxplot(separated_mi_Ys[0].T, sym='')
    ax.set_xticklabels(t_list)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.set_ylabel('DSMI', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax = fig_mi.add_subplot(gs[1:3, 1:2])
    ax.spines[['right', 'top']].set_visible(False)
    ax.boxplot(separated_mi_Ys[1].T, sym='')
    ax.set_xticklabels(t_list)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_mi.add_subplot(gs[1:3, 2:3])
    ax.spines[['right', 'top']].set_visible(False)
    ax.boxplot(single_mi_Ys[0].T, sym='')
    ax.set_xticklabels(t_list)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_mi.add_subplot(gs[1:3, 3:4])
    ax.spines[['right', 'top']].set_visible(False)
    ax.boxplot(single_mi_Ys[1].T, sym='')
    ax.set_xticklabels(t_list)
    ax.set_ylim([ymin, ymax])
    ax.set_xlabel('Diffusion $t$', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)

    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig, bbox_inches='tight')
    plt.close(fig=fig_mi)

import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from sklearn import datasets
from matplotlib.gridspec import GridSpec

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
from diffusion import compute_diffusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/cartoon-MI.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_classes = 3
    cluster_std = 0.3
    N = 1000 * num_classes
    D_max = 30
    num_dim = 15
    num_repetition = 5
    
    D_list = np.linspace(D_max // num_dim, D_max, num_dim, dtype=np.int16)
    t_list = [1,2,3,4,5]


    separated_mi_Ys = [[] for _ in range(len(t_list))]
    single_mi_Ys = [[] for _ in range(len(t_list))]

    for dim in tqdm(D_list):
        embeddings, labels = datasets.make_blobs(
            n_samples=N, n_features=dim, centers=num_classes, cluster_std=cluster_std, random_state=0)
        labels = labels.reshape(N, 1)
        for j, t in enumerate(t_list):
            mi_Y, _, _ = mutual_information_per_class_random_sample(
                embeddings=embeddings,
                labels=labels,
                H_ZgivenY_map=None,
                sigma=args.gaussian_kernel_sigma,
                vne_t=t,
            )
            separated_mi_Ys[j].append(mi_Y)
    
    for dim in tqdm(D_list):
        embeddings, _ = datasets.make_blobs(
            n_samples=N, n_features=dim, centers=1, cluster_std=cluster_std, random_state=0)
        num_per_class = N // num_classes
        labels = np.zeros(N)
        # Randomly assign class labels
        for i in range(1, num_classes+1):
            inds = np.random.choice(np.where(labels == 0)[
                                    0], num_per_class, replace=False)
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
            single_mi_Ys[j].append(mi_Y)
    
    separated_mi_Ys = np.array(separated_mi_Ys)
    single_mi_Ys = np.array(single_mi_Ys)
    print(separated_mi_Ys.shape)

    fig_mi = plt.figure(figsize=(32, 10))
    gs = GridSpec(4, 4, figure=fig_mi)

    for dim, gs_x, gs_y in zip([2, 3], [0, 0], [0, 1]):
        if dim == 3:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y], projection='3d')
        else:
            ax = fig_mi.add_subplot(gs[gs_x, gs_y])
        ax.spines[['right', 'top']].set_visible(False)

        embeddings, labels = datasets.make_blobs(
            n_samples=N, n_features=dim, centers=num_classes, cluster_std=cluster_std, random_state=0)
        if dim == 2:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
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

        embeddings, _ = datasets.make_blobs(
            n_samples=N, n_features=dim, centers=1, cluster_std=cluster_std, random_state=0)
        num_per_class = N // num_classes
        labels = np.zeros(N)
        # Randomly assign class labels
        for i in range(1, num_classes+1):
            inds = np.random.choice(np.where(labels == 0)[
                                    0], num_per_class, replace=False)
            labels[inds] = i
        labels = labels - 1  # class 1,2,3 -> class 0,1,2

        if dim == 2:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           color=color,
                           alpha=0.5)
        elif dim == 3:
            for k, color in enumerate(['tab:blue', 'tab:orange', 'tab:green']):
                inds = np.where(labels == k)
                ax.scatter(embeddings[inds, 0],
                           embeddings[inds, 1],
                           embeddings[inds, 2],
                           color=color,
                           alpha=0.5)
    
    # MI vs. Dims
    ax = fig_mi.add_subplot(gs[1:3, 0:2])
    ax.spines[['right', 'top']].set_visible(False)
    for j in range(len(t_list)):
        ax.plot(D_list,
                separated_mi_Ys[j],
                color=cm.get_cmap('tab10').colors[j])
    ax.legend([
        r'$t$ = %d' % (t) for t in t_list
    ],
        loc='lower right',
        ncol=2)
    # for j in range(len(t_list)):
    #         ax.fill_between(dim_list,
    #                         np.mean(vne_list_uniform[k, j, ...], axis=0) -
    #                         np.std(vne_list_uniform[k, j, ...], axis=0),
    #                         np.mean(vne_list_uniform[k, j, ...], axis=0) +
    #                         np.std(vne_list_uniform[k, j, ...], axis=0),
    #                         color=cm.get_cmap('tab10').colors[j],
    #                         alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_mi.add_subplot(gs[1:3, 2:4])
    ax.spines[['right', 'top']].set_visible(False)
    for j in range(len(t_list)):
        ax.plot(D_list,
                single_mi_Ys[j],
                color=cm.get_cmap('tab10').colors[j])
    ax.legend([
        r'$t$ = %d' % (t) for t in t_list
    ],
        loc='lower right',
        ncol=2)
    # for j in range(len(t_list)):
    #         ax.fill_between(dim_list,
    #                         np.mean(vne_list_uniform[k, j, ...], axis=0) -
    #                         np.std(vne_list_uniform[k, j, ...], axis=0),
    #                         np.mean(vne_list_uniform[k, j, ...], axis=0) +
    #                         np.std(vne_list_uniform[k, j, ...], axis=0),
    #                         color=cm.get_cmap('tab10').colors[j],
    #                         alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    fig_mi.savefig(save_path_fig)
    plt.close(fig=fig_mi)



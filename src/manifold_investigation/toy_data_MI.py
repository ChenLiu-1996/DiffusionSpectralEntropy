import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import random

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/api/')
from dsmi import diffusion_spectral_mutual_information

sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap


def generate_tree(num_points: int = 1000,
                  dim: int = 100,
                  num_branches: int = 10,
                  rand_multiplier: float = 2,
                  random_seed: int = 1):
    '''
    Adapated from
    https://github.com/KrishnaswamyLab/PHATE/blob/8578022459060e8c29e9b37b537a2203e0c7fd6c/Python/phate/tree.py
    '''
    np.random.seed(random_seed)
    branch_length = num_points // num_branches
    M = np.cumsum(-1 + rand_multiplier * np.random.rand(branch_length, dim), 0)
    for _ in range(num_branches - 1):
        ind = np.random.randint(branch_length)
        new_branch = np.cumsum(
            -1 + rand_multiplier * np.random.rand(branch_length, dim), 0)
        M = np.concatenate([M, new_branch + M[ind, :]])

    C = np.array(
        [i // branch_length for i in range(num_branches * branch_length)])

    return M, C


def corrupt_label(labels: np.array,
                  corruption_ratio: float,
                  random_seed: int = 1):
    assert corruption_ratio >= 0 and corruption_ratio <= 1
    if corruption_ratio == 0:
        return labels
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        indices = random.sample(range(len(labels)),
                                k=int(corruption_ratio * len(labels)))
        permuted_indices = np.random.permutation(indices)
        corrupted_labels = labels.copy()
        corrupted_labels[indices] = corrupted_labels[permuted_indices]
        return corrupted_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-MI.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 14

    D = 20
    num_corruption_ratio = 20
    num_repetition = 3

    num_branch_list = [2, 5, 10]  # Currently hard-coded. Has to be 3 items.
    t_list = [1, 2, 3, 5]
    noise_level_list = [1e-2, 1e-1, 5e-1]

    corruption_ratio_list = np.linspace(0, 1, num_corruption_ratio)
    mi_Y_sample_list_tree = [[[[[] for _ in range(num_repetition)]
                               for _ in range(len(t_list))]
                              for _ in range(len(noise_level_list))]
                             for _ in range(len(num_branch_list))]
    mi_Y_shannon_list_tree = [[[[] for _ in range(num_repetition)]
                               for _ in range(len(noise_level_list))]
                              for _ in range(len(num_branch_list))]

    for i in range(num_repetition):
        for corruption_ratio in tqdm(corruption_ratio_list):
            for j, t in enumerate(t_list):
                for k, noise_level in enumerate(noise_level_list):
                    for b, num_branches in enumerate(num_branch_list):

                        tree_data, tree_clusters = generate_tree(
                            dim=D,
                            num_branches=num_branches,
                            num_points=100 * num_branches,
                            random_seed=0)
                        tree_data += noise_level * np.random.uniform(
                            -1, 1, size=tree_data.shape)

                        tree_clusters = corrupt_label(
                            tree_clusters, corruption_ratio=corruption_ratio)

                        mi_Y_sample, _ = diffusion_spectral_mutual_information(
                            embedding_vectors=tree_data,
                            reference_vectors=tree_clusters,
                            reference_discrete=True,
                            t=t,
                            random_seed=i,
                            gaussian_kernel_sigma=args.gaussian_kernel_sigma,
                            chebyshev_approx=False)

                        mi_Y_sample_list_tree[b][k][j][i].append(mi_Y_sample)

                        if j == 0:
                            mi_Y_shannon, _ = diffusion_spectral_mutual_information(
                                embedding_vectors=tree_data,
                                reference_vectors=tree_clusters,
                                classic_shannon_entropy=True,
                                t=t,
                                random_seed=i,
                                gaussian_kernel_sigma=args.
                                gaussian_kernel_sigma,
                                chebyshev_approx=False)
                            mi_Y_shannon_list_tree[b][k][i].append(
                                mi_Y_shannon)

    mi_Y_sample_list_tree = np.array(mi_Y_sample_list_tree)
    mi_Y_shannon_list_tree = np.array(mi_Y_shannon_list_tree)

    fig_mi = plt.figure(figsize=(32, 10))
    gs = GridSpec(4, 9, figure=fig_mi)

    for num_branch_idx, corruption_ratio, gs_y in zip(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [1.0, 0.5, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]):
        ax = fig_mi.add_subplot(gs[0, gs_y], projection='3d')
        ax.spines[['right', 'top']].set_visible(False)

        # Tree.
        tree_data, tree_clusters = generate_tree(
            dim=3,
            num_branches=num_branch_list[num_branch_idx],
            num_points=100 * num_branch_list[num_branch_idx],
            random_seed=0)
        tree_clusters = corrupt_label(tree_clusters,
                                      corruption_ratio=corruption_ratio)
        ax.scatter(tree_data[:, 0],
                   tree_data[:, 1],
                   tree_data[:, 2],
                   c=np.array(cm.get_cmap('tab20').colors)[tree_clusters],
                   alpha=1)

    for num_branch_idx, gs_y_begin in zip([0, 1, 2], [0, 3, 6]):
        ax = fig_mi.add_subplot(gs[1:3, gs_y_begin:gs_y_begin + 3])
        ax.spines[['right', 'top']].set_visible(False)
        linestyle_list = ['solid', 'dashed', 'dotted']
        for j in range(len(t_list)):
            for k in range(len(noise_level_list)):
                ax.plot(corruption_ratio_list,
                        np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j,
                                                      ...],
                                axis=0),
                        color=cm.get_cmap('tab10').colors[j],
                        marker='o',
                        linestyle=linestyle_list[k])
        ax.legend([
            r'$t$ = %d, |noise| = %d%%' % (t, noise * 100) for t in t_list
            for noise in noise_level_list
        ],
                  loc='upper left',
                  ncol=2)
        for j in range(len(t_list)):
            for k in range(len(noise_level_list)):
                ax.fill_between(
                    corruption_ratio_list,
                    np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                            axis=0) -
                    np.std(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                           axis=0),
                    np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                            axis=0) +
                    np.std(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                           axis=0),
                    color=cm.get_cmap('tab10').colors[j],
                    alpha=0.2)
        ax.invert_xaxis()
        ax.axhline(y=0, color='gray', linestyle='-.')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num_branch_idx == 0:
            ax.set_ylabel('DSMI', fontsize=20)

        ax = fig_mi.add_subplot(gs[3:, gs_y_begin:gs_y_begin + 3])
        ax.spines[['right', 'top']].set_visible(False)
        linestyle_list = ['solid', 'dashed', 'dotted']
        for k in range(len(noise_level_list)):
            ax.plot(corruption_ratio_list,
                    np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                            axis=0),
                    color=cm.get_cmap('tab10').colors[-1],
                    marker='o',
                    linestyle=linestyle_list[k])
        ax.legend(
            [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
            loc='upper left',
            ncol=1)
        for k in range(len(noise_level_list)):
            ax.fill_between(
                corruption_ratio_list,
                np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                        axis=0) -
                np.std(mi_Y_shannon_list_tree[num_branch_idx, k, ...], axis=0),
                np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                        axis=0) +
                np.std(mi_Y_shannon_list_tree[num_branch_idx, k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                alpha=0.2)
        ax.invert_xaxis()
        ax.axhline(y=0, color='gray', linestyle='-.')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Label Corruption Ratio', fontsize=25)
        ax.set_ylim([-1.5, 2.5])
        if num_branch_idx == 0:
            ax.set_ylabel('CSMI', fontsize=20)

    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig)
    plt.close(fig=fig_mi)

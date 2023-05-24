import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from phate.tree import gen_dla
import random

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from information import von_neumann_entropy, exact_eigvals
from diffusion import compute_diffusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-DSE-subsample.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_branches = 3
    branch_length = 2000
    num_repetition = 3

    D_list = np.linspace(2, 30, 10, dtype=int)

    t_list = [1, 2, 3, 5]
    ratio_list = [1, 0.5, 0.3, 0.1]
    ses_tree = [[[[] for _ in range(num_repetition)]
                 for _ in range(len(ratio_list))] for _ in range(len(t_list))]

    num_points = num_branches * branch_length

    for i in range(num_repetition):
        for dim in tqdm(D_list):
            tree_data, _ = gen_dla(n_dim=dim,
                                   n_branch=num_branches,
                                   branch_length=branch_length,
                                   sigma=0,
                                   seed=i)

            for j, ratio in enumerate(ratio_list):
                rand_inds = np.array(
                    random.sample(range(num_points),
                                  k=int(num_points * ratio)))
                samples = tree_data[rand_inds, :]

                diffusion_matrix = compute_diffusion_matrix(
                    samples, args.gaussian_kernel_sigma)
                eigenvalues_P = exact_eigvals(diffusion_matrix)

                for k, t in enumerate(t_list):
                    se = von_neumann_entropy(eigenvalues_P, t=t)
                    ses_tree[k][j][i].append(se)

    # Plot
    ses_tree = np.array(ses_tree)
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(t_list)):
        for j in range(len(ratio_list)):
            ax.plot(D_list,
                    np.mean(ses_tree[k, j, ...], axis=0),
                    color=cm.get_cmap('tab10').colors[k],
                    linestyle=linestyle_list[j])
    ax.legend([
        r'$t$ = %d, subsample %d%%' % (t, r * 100) for t in t_list
        for r in ratio_list
    ],
              loc='lower right',
              ncol=2)
    for k in range(len(t_list)):
        for j in range(len(ratio_list)):
            ax.fill_between(D_list,
                            np.mean(ses_tree[k, j, ...], axis=0) -
                            np.std(ses_tree[k, j, ...], axis=0),
                            np.mean(ses_tree[k, j, ...], axis=0) +
                            np.std(ses_tree[k, j, ...], axis=0),
                            color=cm.get_cmap('tab10').colors[k],
                            alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Tree Data Dimension $d$', fontsize=25)
    ax.set_ylabel('Diffusion Spectral Entropy', fontsize=25)

    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig=fig)

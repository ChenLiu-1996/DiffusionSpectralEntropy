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

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from toy_data_diffusion_MI import generate_tree
from information import von_neumann_entropy, exact_eigvals
from diffusion import compute_diffusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-sampling-entropy.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_repetition = 5
    num_branches = 10

    D_list = [20, 40, 60, 80, 100]

    t_list = [1, 2, 3, 5]
    ratio_list = [1.0, 0.5, 0.3, 0.1]
    ses = [[[[] for _ in range(num_repetition)]
            for _ in range(len(ratio_list))]
           for _ in range(len(t_list))]
   
    num_points = 100 * num_branches
    random.seed(0)

    for i in range(num_repetition):
        for dim in tqdm(D_list):
            tree_data, _ = generate_tree(dim=dim,
                                         num_branches=num_branches,
                                         num_points=num_points,
                                         random_seed=args.random_seed)
            for j, ratio in enumerate(ratio_list):
                for k, t in enumerate(t_list):
                    rand_inds = np.array(
                        random.sample(range(num_points),k=int(num_points * ratio)))
                    samples = tree_data[rand_inds, :]

                    diffusion_matrix = compute_diffusion_matrix(samples, args.gaussian_kernel_sigma)
                    eigenvalues_P = exact_eigvals(diffusion_matrix)
                    se = von_neumann_entropy(eigenvalues_P, t=t)
                    ses[k][j][i].append(se)
    

    # Plot
    ses = np.array(ses)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for j in range(len(ratio_list)):
        for k in range(len(t_list)):
            ax.plot(D_list,
                    np.mean(ses[k, j, ...], axis=0),
                    color=cm.get_cmap('tab10').colors[k],
                    linestyle=linestyle_list[k])
    ax.legend([
        r'$t$ = %d, ratio = %d%%' % (t, r * 100) for t in t_list
        for r in ratio_list
    ],
              loc='lower right',
              ncol=2)
    for j in range(len(ratio_list)):
        for k in range(len(t_list)):
            ax.fill_between(D_list,
                            np.mean(ses[k, j, ...], axis=0) -
                            np.std(ses[k, j, ...], axis=0),
                            np.mean(ses[k, j, ...], axis=0) +
                            np.std(ses[k, j, ...], axis=0),
                            color=cm.get_cmap('tab10').colors[k],
                            alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Data Distribution Dimension $d$', fontsize=25)
    ax.set_ylabel('Diffusion Spectral Entropy', fontsize=25)

                    



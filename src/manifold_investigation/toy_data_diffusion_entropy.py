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
from information import von_neumann_entropy, exact_eigvals
from diffusion import compute_diffusion_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=1)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-VNE.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15
    N = 500
    D = 1024
    num_dim = 20
    num_repetition = 5
    matrix_mix_sizes = [400, 420, 440, 460, 480, 500]
    noise_level_list = [0, 5e-2, 1e-1, 2e-1, 3e-1, 5e-1]

    alpha_list = np.linspace(0, 1, num_dim)
    dim_list = np.linspace(D // num_dim, D, num_dim, dtype=np.int16)
    vne_list_matrix = [[[] for _ in range(num_repetition)]
                       for _ in matrix_mix_sizes]
    vne_list_uniform = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]
    vne_list_gaussian = [[[] for _ in range(num_repetition)]
                         for _ in range(len(noise_level_list))]

    for i in range(num_repetition):
        for j, size in enumerate(matrix_mix_sizes):
            matrix_I = np.eye(size)
            matrix_PD = datasets.make_spd_matrix(n_dim=size)
            for alpha in alpha_list:
                matrix = matrix_I * alpha + matrix_PD * (1 - alpha)
                eigenvalues_P = exact_eigvals(matrix)
                vne = von_neumann_entropy(eigenvalues_P)
                vne_list_matrix[j][i].append(vne)

        for dim in tqdm(dim_list):
            for j, noise_level in enumerate(noise_level_list):
                # Uniform distribution over [-1, 1] with distribution dimension == dim.
                embeddings = np.random.uniform(-1, 1, size=(N, D))
                if dim < D:
                    embeddings[:, dim:] = np.random.randn(1)
                embeddings += noise_level * np.random.uniform(
                    -1, 1, size=(N, D))
                diffusion_matrix = compute_diffusion_matrix(embeddings)
                eigenvalues_P = exact_eigvals(diffusion_matrix)
                vne = von_neumann_entropy(eigenvalues_P)
                vne_list_uniform[j][i].append(vne)

                # Normal distribution with distribution dimension == dim.
                embeddings = np.random.randn(N, D)
                if dim < D:
                    embeddings[:, dim:] = np.random.randn(1)
                embeddings += noise_level * np.random.uniform(
                    -1, 1, size=(N, D))
                diffusion_matrix = compute_diffusion_matrix(embeddings)
                eigenvalues_P = exact_eigvals(diffusion_matrix)
                vne = von_neumann_entropy(eigenvalues_P)
                vne_list_gaussian[j][i].append(vne)

    vne_list_matrix = np.array(vne_list_matrix)
    vne_list_uniform = np.array(vne_list_uniform)
    vne_list_gaussian = np.array(vne_list_gaussian)

    # Plot of Diffusion Entropy vs. Dimension.
    fig_vne = plt.figure(figsize=(28, 10))
    gs = GridSpec(4, 9, figure=fig_vne)

    for matrix_type, gs_x, gs_y in zip(['PD', 'I'], [0, 0], [0, 2]):
        ax = fig_vne.add_subplot(gs[gs_x, gs_y])
        matrix_dim = 16
        if matrix_type == 'PD':
            matrix_PD = datasets.make_spd_matrix(n_dim=matrix_dim,
                                                 random_state=9)
            mappable = ax.matshow(matrix_PD, cmap='Blues')
            ax.set_xlabel(r'$n \times n$ Symmetric Matrix', fontsize=20)
            plt.colorbar(mappable, ax=ax, location='left', shrink=0.8)
        else:
            matrix_I = np.eye(matrix_dim)
            mappable = ax.matshow(matrix_I, cmap='Blues')
            ax.set_xlabel(r'$n \times n$ Identity Matrix', fontsize=20)
            plt.colorbar(mappable, ax=ax, shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

    for dim, gs_x, gs_y in zip([1, 2, 3], [0, 0, 0], [3, 4, 5]):
        if dim == 3:
            ax = fig_vne.add_subplot(gs[gs_x, gs_y], projection='3d')
        else:
            ax = fig_vne.add_subplot(gs[gs_x, gs_y])
        ax.spines[['right', 'top']].set_visible(False)

        # Uniform distribution over [-1, 1] with distribution dimension == {1, 2, 3}.
        embeddings = np.random.uniform(-1, 1, size=(N, D))
        if dim < D:
            embeddings[:, dim:] = np.random.randn(1)

        if dim == 1:
            ax.hist(embeddings[:, 0],
                    bins=16,
                    color='mediumblue',
                    edgecolor='white')
        elif dim == 2:
            ax.scatter(embeddings[:, 0],
                       embeddings[:, 1],
                       color='mediumblue',
                       alpha=0.5)
        elif dim == 3:
            ax.scatter(embeddings[:, 0],
                       embeddings[:, 1],
                       embeddings[:, 2],
                       color='mediumblue',
                       alpha=0.5)

    for dim, gs_x, gs_y in zip([1, 2, 3], [0, 0, 0], [6, 7, 8]):
        if dim == 3:
            ax = fig_vne.add_subplot(gs[gs_x, gs_y], projection='3d')
        else:
            ax = fig_vne.add_subplot(gs[gs_x, gs_y])
        ax.spines[['right', 'top']].set_visible(False)

        # Normal distribution with distribution dimension == dim.
        embeddings = np.random.randn(N, D)
        if dim < D:
            embeddings[:, dim:] = np.random.randn(1)

        if dim == 1:
            ax.hist(embeddings[:, 0],
                    bins=16,
                    color='mediumblue',
                    edgecolor='white')
        elif dim == 2:
            ax.scatter(embeddings[:, 0],
                       embeddings[:, 1],
                       color='mediumblue',
                       alpha=0.5)
        elif dim == 3:
            ax.scatter(embeddings[:, 0],
                       embeddings[:, 1],
                       embeddings[:, 2],
                       color='mediumblue',
                       alpha=0.5)

    ax = fig_vne.add_subplot(gs[1:, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    for j, _ in enumerate(matrix_mix_sizes):
        ax.plot(alpha_list,
                np.mean(vne_list_matrix[j, ...], axis=0),
                color=cm.get_cmap('tab10').colors[j])
    ax.legend(['$n$ = %s' % item for item in matrix_mix_sizes],
              loc='lower right',
              ncol=3)
    for j, _ in enumerate(matrix_mix_sizes):
        ax.fill_between(alpha_list,
                        np.mean(vne_list_matrix[j, ...], axis=0) -
                        np.std(vne_list_matrix[j, ...], axis=0),
                        np.mean(vne_list_matrix[j, ...], axis=0) +
                        np.std(vne_list_matrix[j, ...], axis=0),
                        cmap='tab10',
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('Diffusion Spectral Entropy', fontsize=25)
    ax.set_xlabel(r'Weight Coefficient $\alpha$', fontsize=25)

    ax = fig_vne.add_subplot(gs[1:, 3:6])
    ax.spines[['right', 'top']].set_visible(False)
    for j, _ in enumerate(noise_level_list):
        ax.plot(dim_list,
                np.mean(vne_list_uniform[j, ...], axis=0),
                color=cm.get_cmap('tab10').colors[j])
    ax.legend(['|noise| = %d%%' % (item * 100) for item in noise_level_list],
              loc='lower right',
              ncol=3)
    for j, _ in enumerate(noise_level_list):
        ax.fill_between(dim_list,
                        np.mean(vne_list_uniform[j, ...], axis=0) -
                        np.std(vne_list_uniform[j, ...], axis=0),
                        np.mean(vne_list_uniform[j, ...], axis=0) +
                        np.std(vne_list_uniform[j, ...], axis=0),
                        cmap='tab10',
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Data Distribution Dimension $d$', fontsize=25)

    ax = fig_vne.add_subplot(gs[1:, 6:])
    ax.spines[['right', 'top']].set_visible(False)
    for j, _ in enumerate(noise_level_list):
        ax.plot(dim_list,
                np.mean(vne_list_gaussian[j, ...], axis=0),
                color=cm.get_cmap('tab10').colors[j])
    ax.legend(['|noise| = %d%%' % (item * 100) for item in noise_level_list],
              loc='lower right',
              ncol=3)
    for j, _ in enumerate(noise_level_list):
        ax.fill_between(dim_list,
                        np.mean(vne_list_gaussian[j, ...], axis=0) -
                        np.std(vne_list_gaussian[j, ...], axis=0),
                        np.mean(vne_list_gaussian[j, ...], axis=0) +
                        np.std(vne_list_gaussian[j, ...], axis=0),
                        cmap='tab10',
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('Data Distribution Dimension $d$', fontsize=25)

    fig_vne.tight_layout()
    fig_vne.savefig(save_path_fig)
    plt.close(fig=fig_vne)

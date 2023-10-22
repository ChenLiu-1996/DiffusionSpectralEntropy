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
from information import exact_eigvals, von_neumann_entropy, shannon_entropy
from diffusion import compute_diffusion_matrix

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy, adjacency_spectral_entropy
from dsmi import diffusion_spectral_mutual_information


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=1)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-entropy.png' % (save_root)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_compare = 5

    N = 500
    D = 2048
    num_dim = 20
    num_repetition = 3
    matrix_mix_sizes = [64, 128, 256, 512, 1024, 2048]
    t_list = [1, 2, 3, 5]
    noise_level_list = [1e-2, 1e-1, 5e-1, 7e-1] #1e-2, 1e-1,

    alpha_list = np.linspace(0, 1, num_dim)
    dim_list = np.linspace(D // num_dim, D, num_dim, dtype=np.int16)

    vne_list_uniform = [[[[] for _ in range(num_repetition)]
                         for _ in range(len(t_list))]
                        for _ in range(len(noise_level_list))]
    vne_list_gaussian = [[[[] for _ in range(num_repetition)]
                          for _ in range(len(t_list))]
                         for _ in range(len(noise_level_list))]
    se_list_uniform = [[[] for _ in range(num_repetition)]
                       for _ in range(len(noise_level_list))]
    se_list_gaussian = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]
    dmee_list_uniform = [[[] for _ in range(num_repetition)]
                       for _ in range(len(noise_level_list))]
    dmee_list_gaussian = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]
    knn_list_uniform = [[[] for _ in range(num_repetition)]
                       for _ in range(len(noise_level_list))]
    knn_list_gaussian = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]
    gausadj_list_uniform = [[[] for _ in range(num_repetition)]
                       for _ in range(len(noise_level_list))]
    gausadj_list_gaussian = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]
    anisotropic_list_uniform = [[[] for _ in range(num_repetition)]
                       for _ in range(len(noise_level_list))]
    anisotropic_list_gaussian = [[[] for _ in range(num_repetition)]
                        for _ in range(len(noise_level_list))]

    for i in range(num_repetition):
        for dim in tqdm(dim_list):
            for j, t in enumerate(t_list):
                for k, noise_level in enumerate(noise_level_list):
                    # Uniform distribution over [-1, 1] with distribution dimension == dim.
                    embeddings = np.random.uniform(-1, 1, size=(N, D))
                    if dim < D:
                        embeddings[:, dim:] = np.random.randn(1)
                    embeddings += noise_level * np.random.uniform(
                        -1, 1, size=(N, D))
                    diffusion_matrix = compute_diffusion_matrix(embeddings)
                    eigenvalues_P = exact_eigvals(diffusion_matrix)
                    vne = von_neumann_entropy(eigenvalues_P, t=t)
                    vne_list_uniform[k][j][i].append(vne)
                    if j == 0:
                        se = shannon_entropy(embeddings)
                        se_list_uniform[k][i].append(se)
                        
                        dmee = diffusion_spectral_entropy(
                                embedding_vectors=embeddings, 
                                matrix_entry_entropy=True)
                        dmee_list_uniform[k][i].append(dmee)

                        #KNN
                        knn_entropy = adjacency_spectral_entropy(embeddings, use_knn=True)
                        knn_list_uniform[k][i].append(knn_entropy)

                        #Gaussian Adjacency
                        gausadj_entropy = adjacency_spectral_entropy(embeddings, anisotropic=False)
                        gausadj_list_uniform[k][i].append(gausadj_entropy)

                        #Anisotropic Adj
                        anisotropic_entropy = adjacency_spectral_entropy(embeddings, anisotropic=True)
                        anisotropic_list_uniform[k][i].append(anisotropic_entropy)

                    # Normal distribution with distribution dimension == dim.
                    embeddings = np.random.randn(N, D)
                    if dim < D:
                        embeddings[:, dim:] = np.random.randn(1)
                    embeddings += noise_level * np.random.uniform(
                        -1, 1, size=(N, D))
                    diffusion_matrix = compute_diffusion_matrix(embeddings)
                    eigenvalues_P = exact_eigvals(diffusion_matrix)
                    vne = von_neumann_entropy(eigenvalues_P, t=t)
                    vne_list_gaussian[k][j][i].append(vne)
                    if j == 0:
                        se = shannon_entropy(embeddings)
                        se_list_gaussian[k][i].append(se)

                        dmee = diffusion_spectral_entropy(
                                embedding_vectors=embeddings, 
                                matrix_entry_entropy=True)
                        dmee_list_gaussian[k][i].append(dmee)
                        
                        #KNN
                        knn_entropy = adjacency_spectral_entropy(embeddings, use_knn=True)
                        knn_list_gaussian[k][i].append(knn_entropy)

                        #Gaussian Adjacency
                        gausadj_entropy = adjacency_spectral_entropy(embeddings, anisotropic=False)
                        gausadj_list_gaussian[k][i].append(gausadj_entropy)

                        #Anisotropic Adj
                        anisotropic_entropy = adjacency_spectral_entropy(embeddings, anisotropic=True)
                        anisotropic_list_gaussian[k][i].append(anisotropic_entropy)


    vne_list_uniform = np.array(vne_list_uniform)
    vne_list_gaussian = np.array(vne_list_gaussian)
    se_list_uniform = np.array(se_list_uniform)
    se_list_gaussian = np.array(se_list_gaussian)
    dmee_list_uniform = np.array(dmee_list_uniform)
    dmee_list_gaussian = np.array(dmee_list_gaussian)
    knn_list_uniform = np.array(knn_list_uniform)
    knn_list_gaussian = np.array(knn_list_gaussian)
    gausadj_list_uniform = np.array(gausadj_list_uniform)
    gausadj_list_gaussian = np.array(gausadj_list_gaussian)
    anisotropic_list_uniform = np.array(anisotropic_list_uniform)
    anisotropic_list_gaussian = np.array(anisotropic_list_gaussian)

    # Plot of Diffusion Entropy vs. Dimension.
    fig_vne = plt.figure(figsize=(18, (3+num_compare)*2.5))
    gs = GridSpec(3+num_compare, 6, figure=fig_vne)

    for dim, gs_x, gs_y in zip([1, 2, 3], [0, 0, 0], [0, 1, 2]):
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

    for dim, gs_x, gs_y in zip([1, 2, 3], [0, 0, 0], [3, 4, 5]):
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

    ax = fig_vne.add_subplot(gs[1:3, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for j in range(len(t_list)):
        for k in range(len(noise_level_list)):
            ax.plot(dim_list,
                    np.mean(vne_list_uniform[k, j, ...], axis=0),
                    color=cm.get_cmap('tab10').colors[j],
                    marker='o',
                    linestyle=linestyle_list[k])
    ax.legend([
        r'$t$ = %d, |noise| = %d%%' % (t, noise * 100) for t in t_list
        for noise in noise_level_list
    ],
              loc='lower right',
              ncol=2)
    for j in range(len(t_list)):
        for k in range(len(noise_level_list)):
            ax.fill_between(dim_list,
                            np.mean(vne_list_uniform[k, j, ...], axis=0) -
                            np.std(vne_list_uniform[k, j, ...], axis=0),
                            np.mean(vne_list_uniform[k, j, ...], axis=0) +
                            np.std(vne_list_uniform[k, j, ...], axis=0),
                            color=cm.get_cmap('tab10').colors[j],
                            alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_vne.add_subplot(gs[3:4, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(se_list_uniform[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(se_list_uniform[k, ...], axis=0) -
                        np.std(se_list_uniform[k, ...], axis=0),
                        np.mean(se_list_uniform[k, ...], axis=0) +
                        np.std(se_list_uniform[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('DSE', fontsize=25)

    ax = fig_vne.add_subplot(gs[4:5, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(dmee_list_uniform[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(dmee_list_uniform[k, ...], axis=0) -
                        np.std(dmee_list_uniform[k, ...], axis=0),
                        np.mean(dmee_list_uniform[k, ...], axis=0) +
                        np.std(dmee_list_uniform[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('DMEE', fontsize=25)

    #KNN
    ax = fig_vne.add_subplot(gs[5:6, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(knn_list_uniform[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(knn_list_uniform[k, ...], axis=0) -
                        np.std(knn_list_uniform[k, ...], axis=0),
                        np.mean(knn_list_uniform[k, ...], axis=0) +
                        np.std(knn_list_uniform[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('KNN Adj', fontsize=25)
    #Gaussian
    ax = fig_vne.add_subplot(gs[6:7, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(gausadj_list_uniform[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(gausadj_list_uniform[k, ...], axis=0) -
                        np.std(gausadj_list_uniform[k, ...], axis=0),
                        np.mean(gausadj_list_uniform[k, ...], axis=0) +
                        np.std(gausadj_list_uniform[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('Gaussian Adj', fontsize=25)
    #Anisotropic Gaussian
    ax = fig_vne.add_subplot(gs[7:8, 0:3])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(anisotropic_list_uniform[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(anisotropic_list_uniform[k, ...], axis=0) -
                        np.std(anisotropic_list_uniform[k, ...], axis=0),
                        np.mean(anisotropic_list_uniform[k, ...], axis=0) +
                        np.std(anisotropic_list_uniform[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('Anisotropic Gaussian Adj', fontsize=25)

    ax.set_xlabel('Data Distribution Dimension $d$', fontsize=25)


    ax = fig_vne.add_subplot(gs[1:3, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for j in range(len(t_list)):
        for k in range(len(noise_level_list)):
            ax.plot(dim_list,
                    np.mean(vne_list_gaussian[k, j, ...], axis=0),
                    color=cm.get_cmap('tab10').colors[j],
                    marker='o',
                    linestyle=linestyle_list[k])
    ax.legend([
        r'$t$ = %d, |noise| = %d%%' % (t, noise * 100) for t in t_list
        for noise in noise_level_list
    ],
              loc='lower right',
              ncol=2)
    for j in range(len(t_list)):
        for k in range(len(noise_level_list)):
            ax.fill_between(dim_list,
                            np.mean(vne_list_gaussian[k, j, ...], axis=0) -
                            np.std(vne_list_gaussian[k, j, ...], axis=0),
                            np.mean(vne_list_gaussian[k, j, ...], axis=0) +
                            np.std(vne_list_gaussian[k, j, ...], axis=0),
                            color=cm.get_cmap('tab10').colors[j],
                            alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_vne.add_subplot(gs[3:4, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(se_list_gaussian[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(se_list_gaussian[k, ...], axis=0) -
                        np.std(se_list_gaussian[k, ...], axis=0),
                        np.mean(se_list_gaussian[k, ...], axis=0) +
                        np.std(se_list_gaussian[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax = fig_vne.add_subplot(gs[4:5, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(dmee_list_gaussian[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(dmee_list_gaussian[k, ...], axis=0) -
                        np.std(dmee_list_gaussian[k, ...], axis=0),
                        np.mean(dmee_list_gaussian[k, ...], axis=0) +
                        np.std(dmee_list_gaussian[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    #KNN
    ax = fig_vne.add_subplot(gs[5:6, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(knn_list_gaussian[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(knn_list_gaussian[k, ...], axis=0) -
                        np.std(knn_list_gaussian[k, ...], axis=0),
                        np.mean(knn_list_gaussian[k, ...], axis=0) +
                        np.std(knn_list_gaussian[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #Gaussian
    ax = fig_vne.add_subplot(gs[6:7, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(gausadj_list_gaussian[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(gausadj_list_gaussian[k, ...], axis=0) -
                        np.std(gausadj_list_gaussian[k, ...], axis=0),
                        np.mean(gausadj_list_gaussian[k, ...], axis=0) +
                        np.std(gausadj_list_gaussian[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #Anistropic
    ax = fig_vne.add_subplot(gs[7:8, 3:])
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashdot', 'dashed', 'dotted']
    for k in range(len(noise_level_list)):
        ax.plot(dim_list,
                np.mean(anisotropic_list_gaussian[k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[-1],
                marker='o',
                linestyle=linestyle_list[k])
    ax.legend(
        [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
        loc='lower right',
        ncol=3)
    for k in range(len(noise_level_list)):
        ax.fill_between(dim_list,
                        np.mean(anisotropic_list_gaussian[k, ...], axis=0) -
                        np.std(anisotropic_list_gaussian[k, ...], axis=0),
                        np.mean(anisotropic_list_gaussian[k, ...], axis=0) +
                        np.std(anisotropic_list_gaussian[k, ...], axis=0),
                        color=cm.get_cmap('tab10').colors[-1],
                        alpha=0.2)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.set_xlabel('Data Distribution Dimension $d$', fontsize=25)

    fig_vne.tight_layout()
    fig_vne.savefig(save_path_fig)
    plt.close(fig=fig_vne)

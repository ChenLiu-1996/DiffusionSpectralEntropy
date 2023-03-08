import argparse
import os
import sys
from glob import glob

import graphtools as gt
import networkx as nx
import numpy as np
import pandas as pd
import phate
import scipy
import scprep
import seaborn as sns
import yaml
from diffusion_curvature.core import DiffusionMatrix
from diffusion_curvature.laziness import curvature
from matplotlib import pyplot as plt
from scipy.linalg import svd
# from scipy import sparse
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
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


def von_neumann_entropy(eigenvalues, trivial_thr: float = 0.9):
    #NOTE: Shall we use the SVD version?
    # _, eigenvalues, _ = svd(diffusion_matrix)

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the biggest eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues < trivial_thr]

    # Shift the negative eigenvalue(s).
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))


def get_laplacian_extrema(data,
                          n_extrema,
                          knn=10,
                          n_pca=100,
                          subsample=True,
                          big_size: int = 10000):
    '''
    Finds the 'Laplacian extrema' of a dataset.  The first extrema is chosen as
    the point that minimizes the first non-trivial eigenvalue of the Laplacian graph
    on the data.  Subsequent extrema are chosen by first finding the unique non-trivial
    non-negative vector that is zero on all previous extrema while at the same time
    minimizing the Laplacian quadratic form, then taking the argmax of this vector.
    '''

    if subsample and data.shape[0] > big_size:
        data = data[
            np.random.choice(data.shape[0], big_size, replace=False), :]
    G = gt.Graph(data, use_pygsp=True, decay=None, knn=knn, n_pca=n_pca)

    # We need to convert G into a NetworkX graph to use the Tracemin PCG algorithm
    G_nx = nx.convert_matrix.from_scipy_sparse_array(G.W)
    fiedler = nx.linalg.algebraicconnectivity.fiedler_vector(
        G_nx, method='tracemin_pcg')

    # Combinatorial Laplacian gives better results than the normalized Laplacian
    L = nx.laplacian_matrix(G_nx)
    first_extrema = np.argmax(fiedler)
    extrema = [first_extrema]
    extrema_ordered = [first_extrema]

    init_lanczos = fiedler
    init_lanczos = np.delete(init_lanczos, first_extrema)
    for _ in range(n_extrema - 1):
        # Generate the Laplacian submatrix by removing rows/cols for previous extrema
        indices = range(data.shape[0])
        indices = np.delete(indices, extrema)
        ixgrid = np.ix_(indices, indices)
        L_sub = L[ixgrid]

        # Find the smallest eigenvector of our Laplacian submatrix
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(L_sub,
                                                     k=1,
                                                     which='SM',
                                                     v0=init_lanczos)

        # Add it to the sorted and unsorted lists of extrema
        new_extrema = np.argmax(np.abs(eigvecs[:, 0]))
        init_lanczos = eigvecs[:, 0]
        init_lanczos = np.delete(init_lanczos, new_extrema)
        shift = np.searchsorted(extrema_ordered, new_extrema)
        extrema_ordered.insert(shift, new_extrema + shift)
        extrema.append(new_extrema + shift)

    return extrema


def compute_curvature(diffusion_matrix, diffusion_power):
    diffusion_curvatures = curvature(diffusion_matrix,
                                     diffusion_powers=diffusion_power)
    return diffusion_curvatures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument('--seed', help='random seed.', type=int, default=0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))

    seed_everything(args.seed)

    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-*' %
             (config.output_save_path, config.dataset, config.contrastive)))

    save_root = './results_diffusion_characteristics/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_DiffusionEigenvalues = '%s/diffusion-eigenvalues-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_fig_ExtremaPhate = '%s/extrema-PHATE-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_ExtremaEucdist = '%s/extrema-EucDist-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_fig_vonNeumann = '%s/von-Neumann-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_fig_Curvature = '%s/curvature-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_fig_CurvaturePhate = '%s/curvature-PHATE-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    log_path = '%s/log-%s-%s-knn-%s.txt' % (save_root, config.dataset,
                                            config.contrastive, args.knn)

    num_rows = len(embedding_folders)
    fig_DiffusionEigenvalues = plt.figure(figsize=(12, 6 * num_rows))
    fig_ExtremaPhate = plt.figure(figsize=(8, 5 * num_rows))
    fig_CurvaturePhate = plt.figure(figsize=(12, 5 * num_rows))
    fig_ExtremaEucdist = plt.figure(figsize=(8, 5 * num_rows))
    fig_vonNeumann = plt.figure(figsize=(12, 5))
    fig_Curvature = plt.figure(figsize=(8, 10))
    von_neumann_thr_list = [0.5, 0.7, 0.9, 0.95, 0.99]
    x_axis_text, x_axis_value = [], []
    vne_stats_phateP, vne_stats_diffcurP = {}, {}
    curvature_stats_phateP, curvature_stats_diffcurP = [], []

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))
        checkpoint_name = os.path.basename(embedding_folder)
        log(checkpoint_name, log_path)

        labels, embeddings = None, None

        for file in tqdm(files):
            np_file = np.load(file)
            curr_label = np_file['label_true']
            curr_embedding = np_file['embedding']

            if labels is None:
                labels = curr_label[:, None]  # expand dim to [B, 1]
                embeddings = curr_embedding
            else:
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
        '''PHATE embeddings and Diffusion Matrix'''
        save_path_phate_diffusion = '%s/numpy_files/phate-diffusion/phate-diffusion-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_phate_diffusion), exist_ok=True)
        if os.path.exists(save_path_phate_diffusion):
            data_numpy = np.load(save_path_phate_diffusion)
            embedding_phate = data_numpy['embedding_phate']
            diffusion_matrix_phate = data_numpy['diffusion_matrix']
            diffusion_matrix_diffcur = data_numpy['diffusion_matrix_diffcur']
            optimal_t = data_numpy['optimal_t']
            print('Pre-computed phate embeddings and diffusion matrix loaded.')
        else:
            phate_op = phate.PHATE(random_state=args.seed,
                                   n_jobs=1,
                                   n_components=2,
                                   knn=args.knn,
                                   verbose=False)
            embedding_phate = phate_op.fit_transform(embeddings)
            diffusion_matrix_phate = phate_op.graph.diff_op.toarray()
            diffusion_matrix_diffcur = DiffusionMatrix(
                embeddings, kernel_type="adaptive anisotropic", k=args.knn)
            optimal_t = phate_op._find_optimal_t(t_max=100,
                                                 plot=False,
                                                 ax=None)
            with open(save_path_phate_diffusion, 'wb+') as f:
                np.savez(f,
                         embedding_phate=embedding_phate,
                         diffusion_matrix=diffusion_matrix_phate,
                         diffusion_matrix_diffcur=diffusion_matrix_diffcur,
                         optimal_t=optimal_t)
            print('Phate embeddings and diffusion matrix computed.')

        #
        '''Diffusion Eigenvalues'''
        save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
        if os.path.exists(save_path_eigenvalues):
            data_numpy = np.load(save_path_eigenvalues)
            eigenvalues_phateP = data_numpy['eigenvalues_phateP']
            eigenvalues_diffcurP = data_numpy['eigenvalues_diffcurP']
            print('Pre-computed eigenvalues loaded.')
        else:
            eigenvalues_phateP = np.linalg.eigvals(diffusion_matrix_phate)
            eigenvalues_diffcurP = np.linalg.eigvals(diffusion_matrix_diffcur)
            with open(save_path_eigenvalues, 'wb+') as f:
                np.savez(f,
                         eigenvalues_phateP=eigenvalues_phateP,
                         eigenvalues_diffcurP=eigenvalues_diffcurP)
            print('Eigenvalues computed.')

        ax = fig_DiffusionEigenvalues.add_subplot(2 * num_rows, 2, 4 * i + 1)
        ax.set_title('%s (phate knn P matrix)' % checkpoint_name)
        ax.hist(eigenvalues_phateP, color='w', edgecolor='k')
        ax = fig_DiffusionEigenvalues.add_subplot(2 * num_rows, 2, 4 * i + 3)
        sns.boxplot(x=eigenvalues_phateP, color='skyblue', ax=ax)
        ax = fig_DiffusionEigenvalues.add_subplot(2 * num_rows, 2, 4 * i + 2)
        ax.set_title('%s (diffcur adaptive anisotropic P matrix)' %
                     checkpoint_name)
        ax.hist(eigenvalues_diffcurP, color='w', edgecolor='k')
        ax = fig_DiffusionEigenvalues.add_subplot(2 * num_rows, 2, 4 * i + 4)
        sns.boxplot(x=eigenvalues_diffcurP, color='skyblue', ax=ax)
        fig_DiffusionEigenvalues.tight_layout()
        fig_DiffusionEigenvalues.savefig(save_path_fig_DiffusionEigenvalues)

        #
        '''Laplacian Extrema in PHATE coordinates'''
        save_path_extrema = '%s/numpy_files/laplacian-extrema/laplacian-extrema-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_extrema), exist_ok=True)
        if os.path.exists(save_path_extrema):
            data_numpy = np.load(save_path_extrema)
            extrema_inds = data_numpy['extrema_inds']
            print('Pre-computed laplacian extrema loaded.')
        else:
            n_extrema = 10
            extrema_inds = get_laplacian_extrema(data=embeddings,
                                                 n_extrema=n_extrema,
                                                 knn=args.knn)
            with open(save_path_extrema, 'wb+') as f:
                np.savez(f, extrema_inds=extrema_inds)
            print('Laplacian extrema computed.')

        ax = fig_ExtremaPhate.add_subplot(num_rows, 1, i + 1)
        colors = np.empty((N), dtype=object)
        colors.fill('Embedding\nVectors')
        colors[extrema_inds] = 'Laplacian\nExtrema'
        cmap = {
            'Embedding\nVectors': 'gray',
            'Laplacian\nExtrema': 'firebrick'
        }
        sizes = np.empty((N), dtype=int)
        sizes.fill(1)
        sizes[extrema_inds] = 50

        scprep.plot.scatter2d(embedding_phate,
                              c=colors,
                              cmap=cmap,
                              title='%s' % checkpoint_name,
                              legend_anchor=(1.25, 1),
                              ax=ax,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=8,
                              s=sizes)

        fig_ExtremaPhate.tight_layout()
        fig_ExtremaPhate.savefig(save_path_fig_ExtremaPhate)

        #
        '''Laplacian Extrema Euclidean Distances'''
        extrema = embeddings[extrema_inds]
        dist_matrix = pairwise_distances(extrema)
        distances = np.array([
            dist_matrix[i, j] for i in range(len(dist_matrix) - 1)
            for j in range(i + 1, len(dist_matrix))
        ])
        dist_mean = distances.mean()
        dist_std = distances.std()

        ax = fig_ExtremaEucdist.add_subplot(num_rows, 1, i + 1)
        sns.heatmap(dist_matrix, ax=ax)
        ax.set_title('%s  Extrema Euc distance: %.2f \u00B1 %.2f' %
                     (checkpoint_name, dist_mean, dist_std))
        log('Extrema Euc distance: %.2f \u00B1 %.2f' % (dist_mean, dist_std),
            log_path)

        fig_ExtremaEucdist.tight_layout()
        fig_ExtremaEucdist.savefig(save_path_ExtremaEucdist)

        #
        '''von Neumann Entropy'''
        # vne_ref = phate_op._von_neumann_entropy(t_max=optimal_t)[1][0]
        log('von Neumann Entropy (phate knn P matrix): ', log_path)
        for trivial_thr in von_neumann_thr_list:
            vne_phateP = von_neumann_entropy(eigenvalues_phateP,
                                             trivial_thr=trivial_thr)
            log(
                '    removing eigenvalues > %.2f: entropy = %.4f' %
                (trivial_thr, vne_phateP), log_path)

            if trivial_thr not in vne_stats_phateP.keys():
                vne_stats_phateP[trivial_thr] = [vne_phateP]
            else:
                vne_stats_phateP[trivial_thr].append(vne_phateP)

        log('von Neumann Entropy (diffcur adaptive anisotropic P matrix): ',
            log_path)
        for trivial_thr in von_neumann_thr_list:
            vne_diffcurP = von_neumann_entropy(eigenvalues_diffcurP,
                                               trivial_thr=trivial_thr)
            log(
                '    removing eigenvalues > %.2f: entropy = %.4f' %
                (trivial_thr, vne_diffcurP), log_path)

            if trivial_thr not in vne_stats_diffcurP.keys():
                vne_stats_diffcurP[trivial_thr] = [vne_diffcurP]
            else:
                vne_stats_diffcurP[trivial_thr].append(vne_diffcurP)

        x_axis_text.append(checkpoint_name.split('_')[-1])
        if '%' in x_axis_text[-1]:
            x_axis_value.append(int(x_axis_text[-1].split('%')[0]) / 100)
        else:
            x_axis_value.append(x_axis_value[-1] + 0.1)

        #
        '''Diffusion Curvature'''
        save_path_curvature = '%s/numpy_files/diffusion-curvature/diffusion-curvature-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_curvature), exist_ok=True)
        if os.path.exists(save_path_curvature):
            data_numpy = np.load(save_path_curvature)
            curvature_phateP = data_numpy['curvature_phateP']
            curvature_diffcurP = data_numpy['curvature_diffcurP']
            print('Pre-computed curvatures loaded.')
        else:
            curvature_phateP = compute_curvature(diffusion_matrix_phate,
                                                 diffusion_power=optimal_t)
            curvature_diffcurP = compute_curvature(diffusion_matrix_diffcur,
                                                   diffusion_power=optimal_t)
            with open(save_path_curvature, 'wb+') as f:
                np.savez(f,
                         curvature_phateP=curvature_phateP,
                         curvature_diffcurP=curvature_diffcurP)
            print('Curvatures computed.')

        curvature_stats_phateP.append(curvature_phateP)
        curvature_stats_diffcurP.append(curvature_diffcurP)

        log('', log_path)

        #
        '''Diffusion Curvature in PHATE coordinates'''
        ax = fig_CurvaturePhate.add_subplot(num_rows, 2, 2 * i + 1)
        colors = curvature_phateP
        sizes = np.ones((N), dtype=int)
        scprep.plot.scatter2d(
            embedding_phate,
            c=colors,
            cmap='coolwarm',
            title='%s Diffusion Curvature\n(phate knn P matrix)' %
            checkpoint_name,
            legend_anchor=(1.25, 1),
            ax=ax,
            xticks=False,
            yticks=False,
            label_prefix='PHATE',
            fontsize=8,
            s=sizes)
        ax = fig_CurvaturePhate.add_subplot(num_rows, 2, 2 * i + 2)
        colors = curvature_diffcurP
        sizes = np.ones((N), dtype=int)
        scprep.plot.scatter2d(
            embedding_phate,
            c=colors,
            cmap='coolwarm',
            title=
            '%s Diffusion Curvature\n(diffcur adaptive anisotropic P matrix)' %
            checkpoint_name,
            legend_anchor=(1.25, 1),
            ax=ax,
            xticks=False,
            yticks=False,
            label_prefix='PHATE',
            fontsize=8,
            s=sizes)

        fig_CurvaturePhate.tight_layout()
        fig_CurvaturePhate.savefig(save_path_fig_CurvaturePhate)

    ax = fig_vonNeumann.add_subplot(1, 2, 1)
    for trivial_thr in von_neumann_thr_list:
        ax.scatter(x_axis_value, vne_stats_phateP[trivial_thr])
    ax.legend(von_neumann_thr_list, bbox_to_anchor=(1.12, 0.4))
    ax.set_xticks(x_axis_value)
    ax.set_xticklabels(x_axis_text)
    ax.set_title(
        'von Neumann Entropy (at different eigenvalue removal threshold)\n(phate knn P matrix)'
    )
    ax.spines[['right', 'top']].set_visible(False)
    # Plot separately to avoid legend mismatch.
    for trivial_thr in von_neumann_thr_list:
        ax.plot(x_axis_value, vne_stats_phateP[trivial_thr])
    ax = fig_vonNeumann.add_subplot(1, 2, 2)
    for trivial_thr in von_neumann_thr_list:
        ax.scatter(x_axis_value, vne_stats_diffcurP[trivial_thr])
    ax.legend(von_neumann_thr_list, bbox_to_anchor=(1.12, 0.4))
    ax.set_xticks(x_axis_value)
    ax.set_xticklabels(x_axis_text)
    ax.set_title(
        'von Neumann Entropy (at different eigenvalue removal threshold)\n(diffcur adaptive anisotropic P matrix)'
    )
    ax.spines[['right', 'top']].set_visible(False)
    # Plot separately to avoid legend mismatch.
    for trivial_thr in von_neumann_thr_list:
        ax.plot(x_axis_value, vne_stats_diffcurP[trivial_thr])
    fig_vonNeumann.tight_layout()
    fig_vonNeumann.savefig(save_path_fig_vonNeumann)

    ax = fig_Curvature.add_subplot(2, 1, 1)
    df = pd.DataFrame(np.array(curvature_stats_phateP).T, columns=x_axis_value)
    sns.boxplot(data=df, color='skyblue', ax=ax, orient='v')
    ax.set_xticklabels(x_axis_text)
    ax.set_title('Diffusion Curvature Distribution (PHATE knn P matrix)')
    ax = fig_Curvature.add_subplot(2, 1, 2)
    df = pd.DataFrame(np.array(curvature_stats_diffcurP).T,
                      columns=x_axis_value)
    sns.boxplot(data=df, color='skyblue', ax=ax, orient='v')
    ax.set_xticklabels(x_axis_text)
    ax.set_title(
        'Diffusion Curvature Distribution (diffcur adaptive anisotropic P matrix)'
    )
    fig_Curvature.tight_layout()
    fig_Curvature.savefig(save_path_fig_Curvature)

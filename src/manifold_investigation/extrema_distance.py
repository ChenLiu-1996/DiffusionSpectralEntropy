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


def compute_rowcol(n_figures: int):
    '''
    Compute a proper number of rows and columns
    for displaying `n_figures`.
    Ideally, shall be close to a square.
    '''
    num_cols = int(np.sqrt(n_figures))
    if n_figures % num_cols == 0:
        num_rows = n_figures // num_cols
    else:
        num_rows = n_figures // num_cols + 1
    return num_rows, num_cols


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument(
        '--random_seed',
        help='Only enter if you want to override the config!!!',
        type=int,
        default=None)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

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

    # NOTE: Take all the checkpoints for all epochs. Ignore the fixed percentage checkpoints.
    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-%s-seed%s-epoch*' %
             (config.output_save_path, config.dataset, method_str,
              config.model, config.random_seed)))

    save_root = './results_extrema_distance/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_ExtremaPhate = '%s/extrema-PHATE-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    save_path_ExtremaEucdist = '%s/extrema-EucDist-%s-%s-%s-seed%s-knn%s' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    save_path_ExtremaEucdistMedian = '%s/extrema-EucDist-Median-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)

    num_rows, num_cols = compute_rowcol(len(embedding_folders))
    fig_ExtremaPhate = plt.figure(figsize=(8 * num_cols, 5 * num_rows))
    fig_ExtremaEucdist = plt.figure(figsize=(8 * num_cols, 5 * num_rows))
    fig_ExtremaEucdistMedian = plt.figure(figsize=(8, 8))

    epoch_list, extrema_dist_median_list = [], []

    for i, embedding_folder in enumerate(embedding_folders):
        epoch_list.append(
            int(embedding_folder.split('epoch')[-1].split('-valAcc')[0]) + 1)

        files = sorted(glob(embedding_folder + '/*'))
        checkpoint_name = os.path.basename(embedding_folder)

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
        '''PHATE embeddings'''
        save_path_phate = '%s/numpy_files/phate/phate-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_phate), exist_ok=True)
        if os.path.exists(save_path_phate):
            data_numpy = np.load(save_path_phate)
            embedding_phate = data_numpy['embedding_phate']
            print('Pre-computed phate embeddings loaded.')
        else:
            phate_op = phate.PHATE(random_state=config.random_seed,
                                   n_jobs=1,
                                   n_components=2,
                                   knn=args.knn,
                                   verbose=False)
            embedding_phate = phate_op.fit_transform(embeddings)
            with open(save_path_phate, 'wb+') as f:
                np.savez(
                    f,
                    embedding_phate=embedding_phate,
                )
            print('Phate embeddings computed.')

        #
        '''Laplacian Extrema in PHATE coordinates'''
        save_path_extrema = '%s/numpy_files/laplacian-extrema/laplacian-extrema-%s.npz' % (
            save_root, checkpoint_name)
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

        # ax = fig_ExtremaPhate.add_subplot(num_rows, num_cols, i + 1)
        # colors = np.empty((N), dtype=object)
        # colors.fill('Embedding\nVectors')
        # colors[extrema_inds] = 'Laplacian\nExtrema'
        # cmap = {
        #     'Embedding\nVectors': 'gray',
        #     'Laplacian\nExtrema': 'firebrick'
        # }
        # sizes = np.empty((N), dtype=int)
        # sizes.fill(1)
        # sizes[extrema_inds] = 50

        # scprep.plot.scatter2d(embedding_phate,
        #                       c=colors,
        #                       cmap=cmap,
        #                       title='%s' % checkpoint_name,
        #                       legend_anchor=(1.25, 1),
        #                       ax=ax,
        #                       xticks=False,
        #                       yticks=False,
        #                       label_prefix='PHATE',
        #                       fontsize=8,
        #                       s=sizes)

        # fig_ExtremaPhate.tight_layout()
        # fig_ExtremaPhate.savefig(save_path_fig_ExtremaPhate)

        #
        '''Laplacian Extrema Euclidean Distances'''
        extrema = embeddings[extrema_inds]
        dist_matrix = pairwise_distances(extrema)
        distances = np.array([
            dist_matrix[i, j] for i in range(len(dist_matrix) - 1)
            for j in range(i + 1, len(dist_matrix))
        ])
        dist_mean = distances.mean()
        dist_median = np.median(distances)
        dist_std = distances.std()

        extrema_dist_median_list.append(dist_median)

        # ax = fig_ExtremaEucdist.add_subplot(num_rows, num_cols, i + 1)
        # sns.heatmap(dist_matrix, ax=ax)
        # ax.set_title(
        #     '%s  Extrema Euc distance: mean:%.2f median:%.2f std:%.2f' %
        #     (checkpoint_name, dist_mean, dist_median, dist_std))
        # fig_ExtremaEucdist.tight_layout()
        # fig_ExtremaEucdist.savefig(save_path_ExtremaEucdist)

        fig_ExtremaEucdistMedian = plt.figure(figsize=(6, 6))
        ax = fig_ExtremaEucdistMedian.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        # Plot separately to avoid legend mismatch.
        ax.plot(epoch_list, extrema_dist_median_list, c='firebrick')
        fig_ExtremaEucdistMedian.supxlabel('Epochs trained')
        fig_ExtremaEucdistMedian.supylabel(
            'Laplacian extrema Euc dist. median')
        fig_ExtremaEucdistMedian.tight_layout()
        fig_ExtremaEucdistMedian.savefig(save_path_ExtremaEucdistMedian)
        plt.close(fig=fig_ExtremaEucdistMedian)

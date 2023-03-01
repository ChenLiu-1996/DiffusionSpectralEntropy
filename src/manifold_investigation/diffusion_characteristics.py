import argparse
import os
import sys
from glob import glob

import numpy as np
import phate
import scprep
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from scipy import sparse
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


def von_neumann_entropy(data, k: int = 1):
    from scipy.linalg import svd
    _, eigenvalues, _ = svd(data)

    eigenvalues = sorted(eigenvalues)
    # Drop the top k biggest eigenvalues.
    eigenvalues = eigenvalues[k:]

    prob = eigenvalues / np.sum(eigenvalues)
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))


def get_laplacian_extrema(data, n_extrema, knn=10, n_pca=100, subsample=True):
    '''
    Finds the 'Laplacian extrema' of a dataset.  The first extrema is chosen as
    the point that minimizes the first non-trivial eigenvalue of the Laplacian graph
    on the data.  Subsequent extrema are chosen by first finding the unique non-trivial
    non-negative vector that is zero on all previous extrema while at the same time
    minimizing the Laplacian quadratic form, then taking the argmax of this vector.
    '''
    import graphtools as gt
    import networkx as nx
    import scipy

    if subsample and data.shape[0] > 10000:
        data = data[np.random.choice(data.shape[0], 10000, replace=False), :]
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
    for i in range(n_extrema - 1):
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
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))

    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-*' %
             (config.output_save_path, config.dataset, config.contrastive)))

    save_root = './diffusion_PHATE/'
    os.makedirs(save_root, exist_ok=True)
    save_path_1 = '%s/extrema-PHATE-%s-%s.png' % (save_root, config.dataset,
                                                  config.contrastive)
    save_path_2 = '%s/extrema-dist-PHATE-%s-%s.png' % (
        save_root, config.dataset, config.contrastive)
    log_path = '%s/extrema-PHATE-%s-%s.txt' % (save_root, config.dataset,
                                               config.contrastive)

    num_rows = len(embedding_folders)
    fig1 = plt.figure(figsize=(8, 5 * num_rows))
    fig2 = plt.figure(figsize=(8, 5 * num_rows))

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))
        log(os.path.basename(embedding_folder), log_path)

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

        N, D = embeddings.shape

        assert labels.shape[0] == N
        assert labels.shape[1] == 1

        if config.dataset == 'cifar10':
            labels_updated = np.zeros(labels.shape, dtype='object')
            for k in range(N):
                labels_updated[k] = cifar10_int2name[labels[k].item()]
            labels = labels_updated
            del labels_updated

        # PHATE dimensionality reduction.
        phate_op = phate.PHATE(random_state=0,
                               n_jobs=1,
                               n_components=2,
                               verbose=False)
        data_phate = phate_op.fit_transform(embeddings)

        #
        '''von Neumann Entropy'''
        t = phate_op._find_optimal_t(t_max=100, plot=False, ax=None)
        vne_ref = phate_op._von_neumann_entropy(t_max=t)[1][0]
        vne = von_neumann_entropy(phate_op.diff_op)
        print('phate vne: ', vne_ref, 'our vne: ', vne)
        log('von Neumann Entropy: %.4f' % vne, log_path)

        #
        '''Laplacian Extrema'''
        k = 10
        extrema_inds = get_laplacian_extrema(data=embeddings, n_extrema=k)

        ax = fig1.add_subplot(num_rows, 1, i + 1)
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

        scprep.plot.scatter2d(data_phate,
                              c=colors,
                              cmap=cmap,
                              title='%s  von Neumann Entropy: %.3f' %
                              (os.path.basename(embedding_folder), vne),
                              legend_anchor=(1.25, 1),
                              ax=ax,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=8,
                              s=sizes)

        fig1.tight_layout()
        fig1.savefig(save_path_1)

        extrema = embeddings[extrema_inds]
        dist_matrix = pairwise_distances(extrema)
        mean_dist = dist_matrix.sum() / 2 / k

        ax = fig2.add_subplot(num_rows, 1, i + 1)
        sns.heatmap(dist_matrix, ax=ax)
        ax.set_title('%s  Mean extrema Euc distance: %.2f' %
                     (os.path.basename(embedding_folder), mean_dist))
        log('Mean extrema Euc distance: %.2f\n' % mean_dist, log_path)

        fig2.tight_layout()
        fig2.savefig(save_path_2)

import argparse
import os
import sys
from glob import glob

import graphtools as gt
import networkx as nx
import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
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


def LaplacianMatrix(data, knn: int = 10, n_pca: int = 100):
    G = gt.Graph(data, use_pygsp=True, decay=None, knn=knn, n_pca=n_pca)
    G_nx = nx.convert_matrix.from_scipy_sparse_array(G.W)
    L = nx.normalized_laplacian_matrix(G_nx)

    return L.toarray()


def von_neumann_entropy(eigs, trivial_thr: float = 0.9, method: str = 'shift'):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the biggest eigenvalue(s).
    if trivial_thr is not None:
        eigenvalues = eigenvalues[eigenvalues <= trivial_thr]

    if method == 'shift':
        # Shift the negative eigenvalue(s).
        if eigenvalues.min() < 0:
            eigenvalues -= eigenvalues.min()
    elif method == 'truncate':
        # Remove the negative eigenvalue(s).
        eigenvalues = eigenvalues[eigenvalues >= 0]

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))


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

    save_root = './results_laplacian_entropy/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_LaplacianEigenvalues = '%s/laplacian-eigenvalues-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    save_path_fig_vonNeumann = '%s/von-Neumann-%s-%s-knn-%s.png' % (
        save_root, config.dataset, config.contrastive, args.knn)
    log_path = '%s/log-%s-%s-knn-%s.txt' % (save_root, config.dataset,
                                            config.contrastive, args.knn)

    num_rows = len(embedding_folders)
    von_neumann_thr_list = [0.5, 0.8, 1.0, 2.0, 3.0, None]
    x_axis_text, x_axis_value = [], []
    vne_methods = ['truncate', 'shift']
    vne_stats_by_method = {k: {} for k in vne_methods}
    fig_LaplacianEigenvalues = plt.figure(figsize=(8, 6 * num_rows))
    fig_vonNeumann = plt.figure(figsize=(4 * len(vne_methods), 5))

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
        '''Laplacian Matrix'''
        save_path_laplacian = '%s/numpy_files/laplacian/laplacian-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_laplacian), exist_ok=True)
        if os.path.exists(save_path_laplacian):
            data_numpy = np.load(save_path_laplacian)
            laplacian_matrix = data_numpy['laplacian_matrix']
            print('Pre-computed laplacian matrix loaded.')
        else:
            laplacian_matrix = LaplacianMatrix(embeddings, knn=args.knn)
            with open(save_path_laplacian, 'wb+') as f:
                np.savez(f, laplacian_matrix=laplacian_matrix)
            print('Laplacian matrix computed.')

        #
        '''Laplacian Eigenvalues'''
        save_path_eigenvalues = '%s/numpy_files/laplacian-eigenvalues/laplacian-eigenvalues-%s-%s-knn-%s-%s.npz' % (
            save_root, config.dataset, config.contrastive, args.knn,
            checkpoint_name.split('_')[-1])
        os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
        if os.path.exists(save_path_eigenvalues):
            data_numpy = np.load(save_path_eigenvalues)
            eigenvalues_L = data_numpy['eigenvalues_L']
            print('Pre-computed eigenvalues loaded.')
        else:
            eigenvalues_L = np.linalg.eigvals(laplacian_matrix)
            with open(save_path_eigenvalues, 'wb+') as f:
                np.savez(f, eigenvalues_L=eigenvalues_L)
            print('Eigenvalues computed.')

        ax = fig_LaplacianEigenvalues.add_subplot(2 * num_rows, 1, 2 * i + 1)
        ax.set_title('%s (Laplacian matrix)' % checkpoint_name)
        ax.hist(eigenvalues_L, color='w', edgecolor='k')
        ax = fig_LaplacianEigenvalues.add_subplot(2 * num_rows, 1, 2 * i + 2)
        sns.boxplot(x=eigenvalues_L, color='skyblue', ax=ax)
        fig_LaplacianEigenvalues.tight_layout()
        fig_LaplacianEigenvalues.savefig(save_path_fig_LaplacianEigenvalues)

        #
        '''von Neumann Entropy'''
        for vne_method in vne_methods:
            log(
                'von Neumann Entropy (Laplacian matrix)\nmethod: %s: ' %
                vne_method, log_path)
            for trivial_thr in von_neumann_thr_list:
                vne = von_neumann_entropy(eigenvalues_L,
                                          trivial_thr=trivial_thr,
                                          method='%s' % vne_method)
                log(
                    '    removing eigenvalues > %.2f: entropy = %.4f' %
                    (trivial_thr if trivial_thr is not None else np.inf, vne), log_path)

                if trivial_thr not in vne_stats_by_method[vne_method].keys():
                    vne_stats_by_method[vne_method][trivial_thr] = [vne]
                else:
                    vne_stats_by_method[vne_method][trivial_thr].append(vne)

        x_axis_text.append(checkpoint_name.split('_')[-1])
        if '%' in x_axis_text[-1]:
            x_axis_value.append(int(x_axis_text[-1].split('%')[0]) / 100)
        else:
            x_axis_value.append(x_axis_value[-1] + 0.1)

    for subplot_idx, vne_method in enumerate(vne_methods):
        ax = fig_vonNeumann.add_subplot(1, len(vne_methods), subplot_idx + 1)
        for trivial_thr in von_neumann_thr_list:
            ax.scatter(x_axis_value,
                       vne_stats_by_method[vne_method][trivial_thr])
        ax.set_xticks(x_axis_value)
        ax.set_xticklabels(x_axis_text)
        ax.set_title('method for eigenvalues < 0: %s' % vne_method)
        ax.spines[['right', 'top']].set_visible(False)
        # Plot separately to avoid legend mismatch.
        for trivial_thr in von_neumann_thr_list:
            ax.plot(x_axis_value, vne_stats_by_method[vne_method][trivial_thr])
    ax.legend([item if item is not None else 'None' for item in von_neumann_thr_list], bbox_to_anchor=(1.00, 0.48))
    fig_vonNeumann.suptitle(
        'von Neumann Entropy at different eigenvalue removal thresholds')
    fig_vonNeumann.supxlabel('model validation accuracy')
    fig_vonNeumann.tight_layout()
    fig_vonNeumann.savefig(save_path_fig_vonNeumann)

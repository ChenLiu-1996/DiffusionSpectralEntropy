import argparse
import os
import sys
from glob import glob

import numpy as np
import seaborn as sns
import yaml
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
from path_utils import update_config_dirs
from seed import seed_everything
from laplacian_extrema import get_laplacian_extrema

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

    # NOTE: Take the fixed percentage checkpoints.
    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-%s-seed%s-*acc_*' %
             (config.output_save_path, config.dataset, method_str,
              config.model, config.random_seed)))

    save_root = './results_extrema_distance/'
    os.makedirs(save_root, exist_ok=True)
    save_path_ExtremaEucdist = '%s/extrema-EucDist-%s-%s-%s-seed%s' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)

    num_rows, num_cols = compute_rowcol(len(embedding_folders))
    fig_ExtremaEucdist = plt.figure(figsize=(8 * num_cols, 5 * num_rows))

    heatmap_min, heatmap_max = 0, None

    for i, embedding_folder in enumerate(embedding_folders):
        checkpoint_name = os.path.basename(embedding_folder)
        checkpoint_acc = checkpoint_name.split('acc_')[1]

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

        ax = fig_ExtremaEucdist.add_subplot(num_rows, num_cols, i + 1)
        if heatmap_max is None:
            heatmap_max = distances.max() * 1.2
        sns.heatmap(dist_matrix, ax=ax, vmin=heatmap_min, vmax=heatmap_max)
        ax.set_title(
            'Acc: %s  Extrema Euc distance: mean:%.2f median:%.2f std:%.2f' %
            (checkpoint_acc, dist_mean, dist_median, dist_std))
        fig_ExtremaEucdist.tight_layout()
        fig_ExtremaEucdist.savefig(save_path_ExtremaEucdist)

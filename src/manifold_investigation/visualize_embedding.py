import argparse
import os
import sys
from glob import glob

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import phate
import scprep
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from laplacian_extrema import get_laplacian_extrema
from path_utils import update_config_dirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument(
        '--random-seed',
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

    if 'contrastive' in config.keys():
        method_str = config.contrastive
    elif 'bad_method' in config.keys():
        method_str = config.bad_method

    # NOTE: Take the fixed percentage checkpoints.
    embedding_folders = sorted(
        glob(
            '%s/embeddings/%s-%s-%s-seed%s%s-*acc_*' %
            (config.output_save_path, config.dataset, method_str, config.model,
             config.random_seed, '-zeroinit' if config.zero_init else '')))

    save_root = './results_visualize/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig = '%s/visualize-%s-%s-%s-seed%s%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, '-zeroinit' if config.zero_init else '')

    plt.rcParams['font.family'] = 'serif'
    num_cols = len(embedding_folders)
    fig = plt.figure(figsize=(3 * num_cols, 6))

    if config.dataset in ['mnist', 'cifar10', 'stl10']:
        num_classes = 10
    elif config.dataset in ['cifar100']:
        num_classes = 100
    elif config.dataset in ['tinyimagenet']:
        num_classes = 200
    elif config.dataset in ['imagenet']:
        num_classes = 1000
    else:
        raise ValueError('Unsupported value for `config.dataset`: %s.' %
                         config.dataset)

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))
        checkpoint_name = os.path.basename(embedding_folder)
        checkpoint_acc = checkpoint_name.split('acc_')[1]

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

        #
        ''' Laplacian extrema '''
        save_path_extrema = '%s/numpy_files/laplacian-extrema/laplacian-extrema-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_extrema), exist_ok=True)
        if os.path.exists(save_path_extrema):
            data_numpy = np.load(save_path_extrema)
            extrema_inds = data_numpy['extrema_inds']
            print('Pre-computed Laplacian extrema loaded.')
        else:
            n_extrema = num_classes
            extrema_inds = get_laplacian_extrema(data=embeddings,
                                                 n_extrema=n_extrema,
                                                 knn=args.knn)
            with open(save_path_extrema, 'wb+') as f:
                np.savez(f, extrema_inds=extrema_inds)
            print('Laplacian extrema computed.')

        #
        ''' Plotting '''
        #
        ''' PHATE plot, colored by class. '''
        ax = fig.add_subplot(2, num_cols, i + 1)
        phate_op = phate.PHATE(random_state=0,
                               n_jobs=1,
                               n_components=2,
                               t=10,
                               verbose=False)
        data_phate = phate_op.fit_transform(embeddings)
        scprep.plot.scatter2d(data_phate,
                              c=labels,
                              legend=False,
                              ax=ax,
                              title=checkpoint_acc,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=3)

        #
        ''' PHATE plot, annotated by Laplacian extrema. '''
        ax = fig.add_subplot(2, num_cols, i + 1 + num_cols)

        colors_extrema = np.empty((N), dtype=object)
        colors_extrema.fill('Embedding\nVectors')
        colors_extrema[extrema_inds] = 'Laplacian\nExtrema'
        cmap_extrema = {
            'Embedding\nVectors': 'gray',
            'Laplacian\nExtrema': 'firebrick'
        }
        sizes = np.empty((N), dtype=int)
        sizes.fill(1)
        sizes[extrema_inds] = 50

        scprep.plot.scatter2d(data_phate,
                              c=colors_extrema,
                              cmap=cmap_extrema,
                              title=None,
                              legend=False,
                              ax=ax,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=sizes)

        fig.tight_layout()
        fig.savefig(save_path_fig)

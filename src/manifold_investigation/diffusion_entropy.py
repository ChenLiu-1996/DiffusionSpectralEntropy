import argparse
import os
import sys
from glob import glob

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
from diffusion import DiffusionMatrix
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


def von_neumann_entropy(eigs, trivial_thr: float = 0.9, eps: float = 1e-3):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the biggest eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues <= trivial_thr]

    # Shift the negative eigenvalue(s).
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()
    # Remove the close-to-zero eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues >= eps]

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))


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
        glob('%s/embeddings/*%s-%s-%s-seed%s-epoch-*' %
             (config.output_save_path, config.dataset, method_str,
              config.model, config.random_seed)))

    save_root = './results_diffusion_entropy/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_vne = '%s/diffusion-entropy-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    log_path = '%s/log-%s-%s-%s-seed%s-knn%s.txt' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)

    num_rows = len(embedding_folders)
    vne_thr_list = [0.8, 0.9, 0.95, 0.99, 1.00]
    x_axis_value = []
    vne_stats = {}

    for i, embedding_folder in enumerate(embedding_folders):
        x_axis_value.append(int(embedding_folder.split('epoch-')[-1]) + 1)

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
        '''Diffusion Matrix'''
        save_path_diffusion = '%s/numpy_files/diffusion/diffusion-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_diffusion), exist_ok=True)
        if os.path.exists(save_path_diffusion):
            data_numpy = np.load(save_path_diffusion)
            diffusion_matrix = data_numpy['diffusion_matrix']
            print('Pre-computed diffusion matrix loaded.')
        else:
            diffusion_matrix = DiffusionMatrix(embeddings, k=args.knn)
            with open(save_path_diffusion, 'wb+') as f:
                np.savez(f, diffusion_matrix=diffusion_matrix)
            print('Diffusion matrix computed.')

        #
        '''Diffusion Eigenvalues'''
        save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
        if os.path.exists(save_path_eigenvalues):
            data_numpy = np.load(save_path_eigenvalues)
            eigenvalues_P = data_numpy['eigenvalues_P']
            print('Pre-computed eigenvalues loaded.')
        else:
            eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
            with open(save_path_eigenvalues, 'wb+') as f:
                np.savez(f, eigenvalues_P=eigenvalues_P)
            print('Eigenvalues computed.')

        #
        '''von Neumann Entropy'''
        log('von Neumann Entropy (diffcur adaptive anisotropic P matrix): ',
            log_path)
        for trivial_thr in vne_thr_list:
            vne = von_neumann_entropy(eigenvalues_P, trivial_thr=trivial_thr)
            log(
                '    removing eigenvalues > %.2f: entropy = %.4f' %
                (trivial_thr, vne), log_path)

            if trivial_thr not in vne_stats.keys():
                vne_stats[trivial_thr] = [vne]
            else:
                vne_stats[trivial_thr].append(vne)

        fig_vne = plt.figure(figsize=(6, 6))
        ax = fig_vne.add_subplot(1, 1, 1)
        for trivial_thr in vne_thr_list:
            ax.scatter(x_axis_value, vne_stats[trivial_thr])
        ax.set_xticks(x_axis_value)
        ax.spines[['right', 'top']].set_visible(False)
        # Plot separately to avoid legend mismatch.
        for trivial_thr in vne_thr_list:
            ax.plot(x_axis_value, vne_stats[trivial_thr])
        ax.legend(vne_thr_list, bbox_to_anchor=(1.00, 0.48))
        fig_vne.suptitle(
            'von Neumann Entropy at different eigenvalue removal thresholds')
        fig_vne.supxlabel('Epochs trained')
        fig_vne.supylabel('Diffusion Entropy')
        fig_vne.tight_layout()
        fig_vne.savefig(save_path_fig_vne)
        plt.close(fig=fig_vne)

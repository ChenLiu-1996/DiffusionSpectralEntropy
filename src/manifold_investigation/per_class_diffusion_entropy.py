import argparse
import os
import sys
from glob import glob

import numpy as np
import seaborn as sns
import yaml
from diffusion_curvature.core import DiffusionMatrix
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


def von_neumann_entropy(eigs, trivial_thr: float = 0.9):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the biggest eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues <= trivial_thr]

    # Shift the negative eigenvalue(s).
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()

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

    if 'contrastive' in config.keys():
        method_str = config.contrastive
        x_axis_title = 'model validation accuracy'
    elif 'bad_method' in config.keys():
        method_str = config.bad_method
        x_axis_title = 'model train/validation divergence'

    embedding_folders = sorted(
        glob('%s/embeddings/*%s-%s-*' %
             (config.output_save_path, config.dataset, method_str)))

    save_root = './results_diffusion_entropy_PerClass/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_vne = '%s/diffusion-entropy-%s-%s-knn-%s.png' % (
        save_root, config.dataset, method_str, args.knn)
    log_path = '%s/log-%s-%s-knn-%s.txt' % (save_root, config.dataset,
                                            method_str, args.knn)

    num_rows = len(embedding_folders)
    vne_thr_list = [0.8, 0.9, 0.95, 0.99, 1.00]
    x_axis_text, x_axis_value = [], []
    vne_stats = {}
    vne_std = {}
    vne_mean = {}
    fig_DiffusionEigenvalues = plt.figure(figsize=(8, 6 * num_rows))
    fig_vne = plt.figure(figsize=(6, 6))

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

        log('von Neumann Entropy (diffcur adaptive anisotropic P matrix): ',
            log_path)
        # Take the val data by each class and compute entropy
        n_classes = len(np.unique(labels))
        sample_stats = np.zeros((n_classes, len(vne_thr_list)))
        for class_idx in tqdm(np.unique(labels)):
            inds = (labels == class_idx).reshape(-1)
            samples = embeddings[inds, :]

            # Diffusion Matrix
            s_diffusion_matrix = DiffusionMatrix(
                samples, kernel_type="adaptive anisotropic", k=args.knn)
            # Eigenvalues
            s_eigenvalues_P = np.linalg.eigvals(s_diffusion_matrix)
            # Von Neumann Entropy
            for trivial_thr_idx in range(len(vne_thr_list)):
                trivial_thr = vne_thr_list[trivial_thr_idx]
                s_vne = von_neumann_entropy(s_eigenvalues_P,
                                            trivial_thr=trivial_thr)
                sample_stats[class_idx, trivial_thr_idx] = s_vne

        x_axis_text.append(checkpoint_name.split('_')[-1])
        if '%' in x_axis_text[-1]:
            x_axis_value.append(int(x_axis_text[-1].split('%')[0]) / 100)
        else:
            x_axis_value.append(x_axis_value[-1] + 0.1)

        # Compute sample mean, std
        means = np.mean(sample_stats, axis=0).tolist()
        stds = np.std(sample_stats, axis=0).tolist()

        for trivial_thr_idx in range(len(vne_thr_list)):
            trivial_thr = vne_thr_list[trivial_thr_idx]
            std = stds[trivial_thr_idx]
            mean = means[trivial_thr_idx]

            if trivial_thr not in vne_std.keys():
                vne_std[trivial_thr] = [std]
            else:
                vne_std[trivial_thr].append(std)
            if trivial_thr not in vne_mean.keys():
                vne_mean[trivial_thr] = [mean]
            else:
                vne_mean[trivial_thr].append(mean)

            log('    removing samples eigenvalues > %.2f: entropy mean = %.4f, std = %.4f'
                % (trivial_thr, mean, std),
                log_path,
                to_console=False)

    ax = fig_vne.add_subplot(1, 1, 1)
    for trivial_thr in vne_thr_list:
        ax.plot(x_axis_value, vne_mean[trivial_thr])
    ax.set_xticks(x_axis_value)
    ax.set_xticklabels(x_axis_text)
    ax.spines[['right', 'top']].set_visible(False)
    for trivial_thr in vne_thr_list:
        ax.fill_between(
            x_axis_value,
            np.array(vne_mean[trivial_thr]) - np.array(vne_std[trivial_thr]),
            np.array(vne_mean[trivial_thr]) + np.array(vne_std[trivial_thr]),
            alpha=0.2)
    ax.legend(vne_thr_list, bbox_to_anchor=(1.00, 0.48))

    fig_vne.suptitle(
        'von Neumann Entropy at different eigenvalue removal thresholds')
    fig_vne.supxlabel(x_axis_title)
    fig_vne.tight_layout()
    fig_vne.savefig(save_path_fig_vne)

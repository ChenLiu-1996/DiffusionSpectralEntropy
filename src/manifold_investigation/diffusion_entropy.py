import argparse
import os
import sys
from glob import glob

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from characteristics import mutual_information_per_class, von_neumann_entropy, mutual_information
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

    save_root = './results_diffusion_entropy/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_vne = '%s/diffusion-entropy-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    save_path_fig_vne_corr = '%s/diffusion-entropy-corr-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    save_path_fig_mi = '%s/class-mutual-information-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    save_path_fig_mi_corr = '%s/class-mutual-information-corr-%s-%s-%s-seed%s-knn%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)
    log_path = '%s/log-%s-%s-%s-seed%s-knn%s.txt' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)

    num_rows = len(embedding_folders)
    epoch_list, acc_list, vne_list, mi_list, mi_input_list = [], [], [], [], []

    for i, embedding_folder in enumerate(embedding_folders):
        epoch_list.append(
            int(embedding_folder.split('epoch')[-1].split('-valAcc')[0]) + 1)
        acc_list.append(
            float(
                embedding_folder.split('-valAcc')[1].split('-divergence')[0]))

        files = sorted(glob(embedding_folder + '/*'))
        checkpoint_name = os.path.basename(embedding_folder)
        log(checkpoint_name, log_path)

        labels, embeddings, orig_input = None, None, None

        for file in tqdm(files):
            np_file = np.load(file)
            curr_label = np_file['label_true']
            curr_embedding = np_file['embedding']
            curr_input = np_file['image']

            if labels is None:
                labels = curr_label[:, None]  # expand dim to [B, 1]
                embeddings = curr_embedding
                orig_input = curr_input
            else:
                labels = np.vstack((labels, curr_label[:, None]))
                embeddings = np.vstack((embeddings, curr_embedding))
                orig_input = np.vstack((orig_input, curr_input))

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
        '''Diffusion Matrix and Diffusion Eigenvalues'''
        save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
        if os.path.exists(save_path_eigenvalues):
            data_numpy = np.load(save_path_eigenvalues)
            eigenvalues_P = data_numpy['eigenvalues_P']
            print('Pre-computed eigenvalues loaded.')
        else:
            diffusion_matrix = DiffusionMatrix(embeddings, k=args.knn)
            print('Diffusion matrix computed.')
            eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
            # Lower precision to save disk space.
            eigenvalues_P = eigenvalues_P.astype(np.float16)
            with open(save_path_eigenvalues, 'wb+') as f:
                np.savez(f, eigenvalues_P=eigenvalues_P)
            print('Eigenvalues computed.')

        #
        '''Diffusion Entropy'''
        log('von Neumann Entropy (diffcur adaptive anisotropic P matrix): ',
            log_path)
        vne = von_neumann_entropy(eigenvalues_P)
        vne_list.append(vne)
        log('Diffusion Entropy = %.4f' % vne, log_path)

        #
        '''Mutual Information between h_m and Output Class'''
        log('Mutual Information between h_m and Output Class: ', log_path)
        classes_list, classes_cnts = np.unique(labels, return_counts=True)
        vne_by_classes = []
        for class_idx in tqdm(classes_list):
            inds = (labels == class_idx).reshape(-1)
            samples = embeddings[inds, :]

            # Diffusion Matrix
            s_diffusion_matrix = DiffusionMatrix(samples, k=args.knn)
            # Eigenvalues
            s_eigenvalues_P = np.linalg.eigvals(s_diffusion_matrix)
            # Von Neumann Entropy
            s_vne = von_neumann_entropy(s_eigenvalues_P)

            vne_by_classes.append(s_vne)

        mi = mutual_information_per_class(eigenvalues_P,
                                          vne_by_classes,
                                          classes_cnts.tolist(),
                                          unconditioned_entropy=vne)
        mi_list.append(mi)

        #
        '''Mutual Information between h_m and Input'''
        log('Mutual Information between h_m and Input: ', log_path)
        orig_input = np.reshape(orig_input,
                                (N, -1))  # [N, W, H, C] -> [N, W*H*C]
        # MI with input H(h_m) - H(h_m|input)
        mi_input = mutual_information(orig_x=embeddings,
                                      cond_x=orig_input,
                                      knn=args.knn,
                                      class_method='bin',
                                      num_class=100,
                                      orig_entropy=vne)

        mi_input_list.append(mi_input)

        #
        '''Plotting'''
        plt.rcParams['font.family'] = 'serif'

        # Plot of Diffusion Entropy vs. epoch.
        fig_vne = plt.figure(figsize=(20, 20))
        ax = fig_vne.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.scatter(epoch_list, vne_list, c='mediumblue', s=120)
        ax.plot(epoch_list, vne_list, c='mediumblue')
        fig_vne.supylabel('Diffusion Entropy', fontsize=40)
        fig_vne.supxlabel('Epochs Trained', fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        fig_vne.savefig(save_path_fig_vne)
        plt.close(fig=fig_vne)

        # Plot of Diffusion Entropy vs. Val. Acc.
        fig_vne_corr = plt.figure(figsize=(20, 20))
        ax = fig_vne_corr.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.scatter(acc_list,
                   vne_list,
                   facecolors='none',
                   edgecolors='mediumblue',
                   s=500,
                   linewidths=5)
        fig_vne_corr.supylabel('Diffusion Entropy', fontsize=40)
        fig_vne_corr.supxlabel('Downstream Classification Accuracy',
                               fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        # Display correlation.
        if len(acc_list) > 1:
            fig_vne_corr.suptitle(
                'Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f)' %
                (pearsonr(acc_list, vne_list)[0], pearsonr(
                    acc_list, vne_list)[1], spearmanr(acc_list, vne_list)[0],
                 spearmanr(acc_list, vne_list)[1]),
                fontsize=40)
        fig_vne_corr.savefig(save_path_fig_vne_corr)
        plt.close(fig=fig_vne_corr)

        # Plot of Mutual Information vs. epoch.
        fig_mi = plt.figure(figsize=(20, 20))
        ax = fig_mi.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        # MI wrt Output
        ax.scatter(epoch_list, mi_list, c='mediumblue', s=120)
        ax.plot(epoch_list, mi_list, c='mediumblue')
        # MI wrt Input
        ax.scatter(epoch_list, mi_input_list, c='mediumgreen', s=120)
        ax.plot(epoch_list, mi_input_list, c='mediumgreen')
        ax.legend(['I(h_m;Y)', 'I(h_m;X)'], bbox_to_anchor=(1.00, 0.48))
        fig_mi.supylabel('Mutual Information', fontsize=40)
        fig_mi.supxlabel('Epochs Trained', fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        fig_mi.savefig(save_path_fig_mi)
        plt.close(fig=fig_mi)

        # Plot of Mutual Information vs. Val. Acc.
        fig_mi_corr = plt.figure(figsize=(20, 20))
        ax = fig_mi_corr.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.scatter(acc_list,
                   mi_list,
                   facecolors='none',
                   edgecolors='mediumblue',
                   s=500,
                   linewidths=5)
        ax.scatter(acc_list,
                   mi_input_list,
                   facecolors='none',
                   edgecolors='mediumgreen',
                   s=500,
                   linewidths=5)
        ax.legend(['I(h_m;Y)', 'I(h_m;X)'], bbox_to_anchor=(1.00, 0.48))
        fig_mi_corr.supylabel('Mutual Information', fontsize=40)
        fig_mi_corr.supxlabel('Downstream Classification Accuracy',
                              fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        # Display correlation.
        if len(acc_list) > 1:
            fig_mi_corr.suptitle(
                'I(h_m;Y) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);'
                % (pearsonr(acc_list, mi_list)[0], pearsonr(
                    acc_list, mi_list)[1], spearmanr(acc_list, mi_list)[0],
                   spearmanr(acc_list, mi_list)[1]) +
                'I(h_m;X) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);'
                % (pearsonr(acc_list, mi_input_list)[0],
                   pearsonr(acc_list, mi_input_list)[1],
                   spearmanr(acc_list, mi_input_list)[0],
                   spearmanr(acc_list, mi_input_list)[1]),
                fontsize=20)
        fig_mi_corr.savefig(save_path_fig_mi_corr)
        plt.close(fig=fig_mi_corr)

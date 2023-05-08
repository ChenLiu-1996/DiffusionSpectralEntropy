import argparse
import os
import sys
from glob import glob

import numpy as np
import yaml
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from typing import Dict, Iterable
import random

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from characteristics import approx_eigvals, mutual_information_per_class_random_sample, von_neumann_entropy, mutual_information, comp_diffusion_embedding
from diffusion import compute_diffusion_matrix
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


def plot_figures(data_arrays: Dict[str, Iterable],
                 save_paths_fig: Dict[str, str]) -> None:

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    # Plot of Diffusion Entropy vs. epoch.
    fig_vne = plt.figure(figsize=(20, 20))
    ax = fig_vne.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['epoch'], data_arrays['vne'], c='mediumblue', s=120)
    ax.plot(data_arrays['epoch'], data_arrays['vne'], c='mediumblue')
    fig_vne.supylabel('Diffusion Entropy', fontsize=40)
    fig_vne.supxlabel('Epochs Trained', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    fig_vne.savefig(save_paths_fig['fig_vne'])
    plt.close(fig=fig_vne)

    # Plot of Diffusion Entropy vs. Val. Acc.
    fig_vne_corr = plt.figure(figsize=(20, 20))
    ax = fig_vne_corr.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['acc'],
               data_arrays['vne'],
               c='mediumblue',
               alpha=0.5,
               s=300)
    fig_vne_corr.supylabel('Diffusion Entropy', fontsize=40)
    fig_vne_corr.supxlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    # Display correlation.
    if len(data_arrays['acc']) > 1:
        fig_vne_corr.suptitle(
            'Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f)' %
            (pearsonr(data_arrays['acc'], data_arrays['vne'])[0],
             pearsonr(data_arrays['acc'], data_arrays['vne'])[1],
             spearmanr(data_arrays['acc'], data_arrays['vne'])[0],
             spearmanr(data_arrays['acc'], data_arrays['vne'])[1]),
            fontsize=40)
    fig_vne_corr.savefig(save_paths_fig['fig_vne_corr'])
    plt.close(fig=fig_vne_corr)

    # Plot of Mutual Information vs. epoch.
    fig_mi = plt.figure(figsize=(20, 20))
    ax = fig_mi.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    # MI wrt Output
    ax.plot(data_arrays['epoch'], data_arrays['mi'], c='mediumblue')
    # MI wrt Input
    # ax.plot(data_arrays['epoch'], data_arrays['mi_input'], c='green')
    # ax.plot(data_arrays['epoch'], data_arrays['mii_spectral'], c='red')
    ax.legend(['I(z;Y)', 'I(z;X)', 'I(z;X) spectral'],
              bbox_to_anchor=(1.00, 0.48))
    ax.scatter(data_arrays['epoch'], data_arrays['mi'], c='mediumblue', s=120)
    # ax.scatter(data_arrays['epoch'], data_arrays['mi_input'], c='green', s=120)
    # ax.scatter(data_arrays['epoch'],
    #            data_arrays['mii_spectral'],
    #            c='red',
    #            s=120)
    fig_mi.supylabel('Mutual Information', fontsize=40)
    fig_mi.supxlabel('Epochs Trained', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    fig_mi.savefig(save_paths_fig['fig_mi'])
    plt.close(fig=fig_mi)

    # Plot of Mutual Information vs. Val. Acc.
    fig_mi_corr = plt.figure(figsize=(20, 20))
    ax = fig_mi_corr.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['acc'],
               data_arrays['mi'],
               c='mediumblue',
               alpha=0.5,
               s=300)
    # ax.scatter(data_arrays['acc'],
    #            data_arrays['mi_input'],
    #            c='green',
    #            alpha=0.5,
    #            s=300,
    #            linewidths=5)
    # ax.scatter(data_arrays['acc'],
    #            data_arrays['mii_spectral'],
    #            c='red',
    #            alpha=0.5,
    #            s=300,
    #            linewidths=5)
    ax.legend(['I(z;Y)', 'I(z;X)', 'I(z;X) spectral'],
              bbox_to_anchor=(1.00, 0.48))
    fig_mi_corr.supylabel('Mutual Information', fontsize=40)
    fig_mi_corr.supxlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # # Display correlation.
    # if len(data_arrays['acc']) > 1:
    #     fig_mi_corr.suptitle(
    #         'I(z;Y) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);\n'
    #         % (pearsonr(data_arrays['acc'], data_arrays['mi'])[0],
    #            pearsonr(data_arrays['acc'], data_arrays['mi'])[1],
    #            spearmanr(data_arrays['acc'], data_arrays['mi'])[0],
    #            spearmanr(data_arrays['acc'], data_arrays['mi'])[1]),  #+
    #         # '\nI(z;X) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);\n'
    #         # % (pearsonr(data_arrays['acc'], data_arrays['mi_input'])[0],
    #         #    pearsonr(data_arrays['acc'], data_arrays['mi_input'])[1],
    #         #    spearmanr(data_arrays['acc'], data_arrays['mi_input'])[0],
    #         #    spearmanr(data_arrays['acc'], data_arrays['mi_input'])[1]),  #+
    #         # '\nSpectral I(z;X) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);'
    #         # % (pearsonr(data_arrays['acc'], data_arrays['mii_spectral'])[0],
    #         #    pearsonr(data_arrays['acc'], data_arrays['mii_spectral'])[1],
    #         #    spearmanr(data_arrays['acc'], data_arrays['mii_spectral'])[0],
    #         #    spearmanr(data_arrays['acc'], data_arrays['mii_spectral'])[1]),
    #         fontsize=40)
    # fig_mi_corr.savefig(save_paths_fig['fig_mi_corr'])
    # plt.close(fig=fig_mi_corr)

    return


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

    save_paths_fig = {
        'fig_vne':
        '%s/diffusion-entropy-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn),
        'fig_vne_corr':
        '%s/diffusion-entropy-corr-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn),
        'fig_mi':
        '%s/class-mutual-information-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn),
        'fig_mi_corr':
        '%s/class-mutual-information-corr-%s-%s-%s-seed%s-knn%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed, args.knn)
    }

    save_path_final_npy = '%s/numpy_files/figure-data-%s-%s-%s-seed%s-knn%s.npy' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)

    log_path = '%s/log-%s-%s-%s-seed%s-knn%s.txt' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed, args.knn)

    os.makedirs(os.path.dirname(save_path_final_npy), exist_ok=True)
    if os.path.exists(save_path_final_npy):
        data_numpy = np.load(save_path_final_npy)
        data_arrays = {
            'epoch': data_numpy['epoch'],
            'acc': data_numpy['acc'],
            'vne': data_numpy['vne'],
            'mi': data_numpy['mi'],
            # 'mi_input': data_numpy['mi_input'],
            # 'mii_spectral': data_numpy['mii_spectral'],
        }
        plot_figures(data_arrays=data_arrays, save_paths_fig=save_paths_fig)

    else:
        epoch_list, acc_list, vne_list, mi_list, mi_input_list, mii_spectral_list = [], [], [], [], [], []

        for i, embedding_folder in enumerate(embedding_folders):
            epoch_list.append(
                int(embedding_folder.split('epoch')[-1].split('-valAcc')[0]) +
                1)
            acc_list.append(
                float(
                    embedding_folder.split('-valAcc')[1].split('-divergence')
                    [0]))

            files = sorted(glob(embedding_folder + '/*'))
            checkpoint_name = os.path.basename(embedding_folder)
            log(checkpoint_name, log_path)

            labels, embeddings, orig_input = None, None, None

            for file in tqdm(files):
                np_file = np.load(file)
                curr_input = np_file['image']
                curr_label = np_file['label_true']
                curr_embedding = np_file['embedding']

                if labels is None:
                    orig_input = curr_input
                    labels = curr_label[:, None]  # expand dim to [B, 1]
                    embeddings = curr_embedding
                else:
                    orig_input = np.vstack((orig_input, curr_input))
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
            '''Diffusion Matrix and Diffusion Eigenvalues'''
            save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-%s.npz' % (
                save_root, checkpoint_name)
            os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
            if os.path.exists(save_path_eigenvalues):
                data_numpy = np.load(save_path_eigenvalues)
                eigenvalues_P = data_numpy['eigenvalues_P']
                print('Pre-computed eigenvalues loaded.')
            else:
                diffusion_matrix = compute_diffusion_matrix(embeddings,
                                                            k=args.knn)
                print('Diffusion matrix computed.')
                eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
                # eigenvalues_P = approx_eigvals(diffusion_matrix)

                # Lower precision to save disk space.
                eigenvalues_P = eigenvalues_P.astype(np.float16)
                with open(save_path_eigenvalues, 'wb+') as f:
                    np.savez(f, eigenvalues_P=eigenvalues_P)
                print('Eigenvalues computed.')

            #
            '''Diffusion Entropy'''
            log(
                'von Neumann Entropy (diffcur adaptive anisotropic P matrix): ',
                log_path)
            vne = von_neumann_entropy(eigenvalues_P)
            vne_list.append(vne)
            log('Diffusion Entropy = %.4f' % vne, log_path)

            #
            '''Mutual Information between z and Output Class'''
            log('Mutual Information between z and Output Class: ', log_path)
            classes_list, classes_cnts = np.unique(labels, return_counts=True)
            vne_by_classes = []
            for class_idx in tqdm(classes_list):
                inds = (labels == class_idx).reshape(-1)
                samples = embeddings[inds, :]

                # Diffusion Matrix
                s_diffusion_matrix = compute_diffusion_matrix(samples,
                                                              k=args.knn)
                # Eigenvalues
                s_eigenvalues_P = np.linalg.eigvals(s_diffusion_matrix)
                # Von Neumann Entropy
                s_vne = von_neumann_entropy(s_eigenvalues_P)

                vne_by_classes.append(s_vne)

            # mi = mutual_information_per_class(eigenvalues_P,
            #                                   vne_by_classes,
            #                                   classes_cnts.tolist(),
            #                                   unconditioned_entropy=vne)
            mi = mutual_information_per_class_random_sample(
                embeddings=embeddings, labels=labels, knn=args.knn)
            mi_list.append(mi)
            log('MI between z and Output = %.4f' % mi, log_path)

            #
            '''Mutual Information between z and Input'''
            # log('Mutual Information between z and Input: ', log_path)
            # orig_input = np.reshape(orig_input,
            #                         (N, -1))  # [N, W, H, C] -> [N, W*H*C]
            # # MI with input H(z) - H(z|input)
            # mi_input, mi_cond, cond_classes_nums = mutual_information(
            #     orig_x=embeddings,
            #     cond_x=orig_input,
            #     knn=args.knn,
            #     class_method='bin',
            #     num_digit=2,
            #     orig_entropy=vne)

            # mi_input_list.append(mi_input)
            # log(
            #     'MI between z and Input = %.4f, Cond Entropy = %.4f, Cond Classes Num: %d '
            #     % (mi_input, mi_cond, cond_classes_nums), log_path)
            #
            '''(Spectral Bin) Mutual Information between z and Input'''
            # log('(Spectral Bin) Mutual Information between z and Input: ',
            #     log_path)
            # # Diffusion embeddings of orig_input
            # save_path_diff_embed = '%s/numpy_files/diffusion-embeddings/%s.npz' % (
            #     save_root, config.dataset)
            # os.makedirs(os.path.dirname(save_path_diff_embed), exist_ok=True)
            # if os.path.exists(save_path_diff_embed):
            #     diff_embed = np.load(save_path_diff_embed)['diff_embed']
            #     print(
            #         'Pre-computed original data diffusion embeddings loaded.')
            # else:
            #     diff_embed = comp_diffusion_embedding(orig_input, knn=args.knn)
            #     print('Original data diffusion embeddings computed.')
            #     diff_embed = diff_embed.astype(np.float16)
            #     with open(save_path_diff_embed, 'wb+') as f:
            #         np.savez(f, diff_embed=diff_embed)

            # mii_spectral, mi_cond_spectral, cond_classes_nums_spectral = mutual_information(
            #     orig_x=embeddings,
            #     cond_x=orig_input,
            #     knn=args.knn,
            #     class_method='spectral_bin',
            #     num_digit=2,
            #     num_spectral=None,
            #     diff_embed=diff_embed,
            #     orig_entropy=vne)

            # mii_spectral_list.append(mii_spectral)
            # log(
            #     '(Spectral Bin) MI between z and Input = %.4f, Cond Entropy = %.4f, Cond Classes Num: %d '
            #     % (mii_spectral, mi_cond_spectral, cond_classes_nums_spectral),
            #     log_path)

            # Plotting
            data_arrays = {
                'epoch': epoch_list,
                'acc': acc_list,
                'vne': vne_list,
                'mi': mi_list,
                # 'mi_input': mi_input_list,
                # 'mii_spectral': mii_spectral_list,
            }
            plot_figures(data_arrays=data_arrays,
                         save_paths_fig=save_paths_fig)

        with open(save_path_final_npy, 'wb+') as f:
            np.savez(f,
                     epoch=np.array(epoch_list),
                     acc=np.array(acc_list),
                     vne=np.array(vne_list),
                     mi=np.array(mi_list))
            #  mi_input=np.array(mi_input_list)
            #  mii_spectral=np.array(mii_spectral_list))

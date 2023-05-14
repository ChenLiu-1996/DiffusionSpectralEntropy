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
from information import approx_eigvals, exact_eigvals, exact_eig, mi_fourier, fourier_entropy, von_neumann_entropy, shannon_entropy, mutual_information, comp_diffusion_embedding, mutual_information_per_class_append
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
    fig_entropy = plt.figure(figsize=(20, 20))
    ax = fig_entropy.add_subplot(1, 1, 1)
    ax_secondary = ax.twinx()
    ax.spines[['right', 'top']].set_visible(False)
    ax_secondary.spines[['left', 'top']].set_visible(False)
    ln1 = ax.plot(data_arrays['epoch'], data_arrays['se'], c='grey')
    ax.scatter(data_arrays['epoch'], data_arrays['se'], c='grey', s=120)
    ln2 = ax_secondary.plot(data_arrays['epoch'],
                            data_arrays['vne'],
                            c='mediumblue')
    ax_secondary.scatter(data_arrays['epoch'],
                         data_arrays['vne'],
                         c='mediumblue',
                         s=120)
    lns = ln1 + ln2
    ax.legend(lns, ['Shannon entropy', 'spectral von Neumann entropy'])
    fig_entropy.supylabel('Diffusion Entropy', fontsize=40)
    fig_entropy.supxlabel('Epochs Trained', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax_secondary.tick_params(axis='both', which='major', labelsize=30)
    fig_entropy.savefig(save_paths_fig['fig_entropy'])
    plt.close(fig=fig_entropy)

    # Plot of Diffusion Entropy vs. Val. Acc.
    fig_entropy_corr = plt.figure(figsize=(20, 20))
    ax = fig_entropy_corr.add_subplot(1, 1, 1)
    ax_secondary = ax.twinx()
    ax.spines[['right', 'top']].set_visible(False)
    ax_secondary.spines[['left', 'top']].set_visible(False)
    ln1 = ax.scatter(data_arrays['acc'],
                     data_arrays['se'],
                     c='grey',
                     alpha=0.5,
                     s=300)
    ln2 = ax_secondary.scatter(data_arrays['acc'],
                               data_arrays['vne'],
                               c='mediumblue',
                               alpha=0.5,
                               s=300)
    ln3 = ax_secondary.scatter(data_arrays['acc'],
                               data_arrays['vne_random'],
                               c='green',
                               alpha=0.5,
                               s=300)
    lns = [ln1] + [ln2] + [ln3]
    ax.legend(lns, ['Shannon entropy', 'Fourier entropy using full signal', 'Fourier entropy using random signal'])
    fig_entropy_corr.supylabel('Diffusion Entropy', fontsize=40)
    fig_entropy_corr.supxlabel('Downstream Classification Accuracy',
                               fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax_secondary.tick_params(axis='both', which='major', labelsize=30)
    # Display correlation.
    if len(data_arrays['acc']) > 1:
        fig_entropy_corr.suptitle(
            'VNE_FULL Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f)' %
            (pearsonr(data_arrays['acc'], data_arrays['vne'])[0],
             pearsonr(data_arrays['acc'], data_arrays['vne'])[1],
             spearmanr(data_arrays['acc'], data_arrays['vne'])[0],
             spearmanr(data_arrays['acc'], data_arrays['vne'])[1]) + 
             'VNE_RANDOM Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f)' %
            (pearsonr(data_arrays['acc'], data_arrays['vne_random'])[0],
             pearsonr(data_arrays['acc'], data_arrays['vne_random'])[1],
             spearmanr(data_arrays['acc'], data_arrays['vne_random'])[0],
             spearmanr(data_arrays['acc'], data_arrays['vne_random'])[1]),
            fontsize=40)
    fig_entropy_corr.savefig(save_paths_fig['fig_entropy_corr'])
    plt.close(fig=fig_entropy_corr)

    # Plot of Mutual Information vs. epoch.
    fig_mi = plt.figure(figsize=(20, 20))
    ax = fig_mi.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    # MI wrt Output
    ax.plot(data_arrays['epoch'], data_arrays['mi'], c='mediumblue')
    ax.plot(data_arrays['epoch'], data_arrays['mi_sample'], c='green')
    # MI wrt Input
    # ax.plot(data_arrays['epoch'], data_arrays['mi_X'], c='green')
    # ax.plot(data_arrays['epoch'], data_arrays['mi_X_spectral'], c='red')
    ax.legend(['I(z;Y) full', 'I(z;X) sample'],
              bbox_to_anchor=(1.00, 0.48))
    ax.scatter(data_arrays['epoch'],
               data_arrays['mi_Y'],
               c='mediumblue',
               s=120)
    ax.scatter(data_arrays['epoch'], data_arrays['mi_sample'], c='green', s=120)
    # ax.scatter(data_arrays['epoch'],
    #            data_arrays['mi_X_spectral'],
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
    ax.scatter(data_arrays['acc'],
               data_arrays['mi_sample'],
               c='green',
               alpha=0.5,
               s=300,
               linewidths=5)
    # ax.scatter(data_arrays['acc'],
    #            data_arrays['mi_X_spectral'],
    #            c='red',
    #            alpha=0.5,
    #            s=300,
    #            linewidths=5)
    ax.legend(['I(z;Y) full', 'I(z;Y) sample'],
              bbox_to_anchor=(1.00, 0.48))
    fig_mi_corr.supylabel('Mutual Information', fontsize=40)
    fig_mi_corr.supxlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # # Display correlation.
    if len(data_arrays['acc']) > 1:
        fig_mi_corr.suptitle(
            'I(z;Y) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);\n'
            % (pearsonr(data_arrays['acc'], data_arrays['mi'])[0],
               pearsonr(data_arrays['acc'], data_arrays['mi'])[1],
               spearmanr(data_arrays['acc'], data_arrays['mi'])[0],
               spearmanr(data_arrays['acc'], data_arrays['mi'])[1]), +
            'I(z;Y) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);\n'
            % (pearsonr(data_arrays['acc'], data_arrays['mi_sample'])[0],
               pearsonr(data_arrays['acc'], data_arrays['mi_sample'])[1],
               spearmanr(data_arrays['acc'], data_arrays['mi_sample'])[0],
               spearmanr(data_arrays['acc'], data_arrays['mi_sample'])[1]),  # +
            # '\nI(z;X) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);\n'
            # % (pearsonr(data_arrays['acc'], data_arrays['mi_X'])[0],
            #    pearsonr(data_arrays['acc'], data_arrays['mi_X'])[1],
            #    spearmanr(data_arrays['acc'], data_arrays['mi_X'])[0],
            #    spearmanr(data_arrays['acc'], data_arrays['mi_X'])[1]),  #+
            # '\nSpectral I(z;X) Pearson R: %.3f (p = %.4f), Spearman R: %.3f (p = %.4f);'
            # % (pearsonr(data_arrays['acc'], data_arrays['mi_X_spectral'])[0],
            #    pearsonr(data_arrays['acc'], data_arrays['mi_X_spectral'])[1],
            #    spearmanr(data_arrays['acc'], data_arrays['mi_X_spectral'])[0],
            #    spearmanr(data_arrays['acc'], data_arrays['mi_X_spectral'])[1]),
            fontsize=40)
    fig_mi_corr.savefig(save_paths_fig['fig_mi_corr'])
    plt.close(fig=fig_mi_corr)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    parser.add_argument(
        '--chebyshev',
        action='store_true',
        help='Chebyshev approximation instead of full eigendecomposition.')
    parser.add_argument(
        '--random-seed',
        help='Only enter if you want to override the config!!!',
        type=int,
        default=None)
    parser.add_argument(
        '--noise_eigval_thr',
        help='noise_eigval_thr',
        type=float,
        default=1e-3)
    parser.add_argument(
        '--num_repetitions',
        help='random num_repetitions',
        type=int,
        default=10)
    parser.add_argument(
        '--topk',
        help='topk comps',
        type=int,
        default=100)
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

    save_root = './fourier_diffusion_entropy/'
    os.makedirs(save_root, exist_ok=True)

    save_paths_fig = {
        'fig_entropy':
        '%s/diffusion-entropy-%s-%s-%s-seed%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed),
        'fig_entropy_corr':
        '%s/diffusion-entropy-corr-%s-%s-%s-seed%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed),
        'fig_mi':
        '%s/class-mutual-information-%s-%s-%s-seed%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed),
        'fig_mi_corr':
        '%s/class-mutual-information-corr-%s-%s-%s-seed%s.png' %
        (save_root, config.dataset, method_str, config.model,
         config.random_seed)
    }

    save_path_final_npy = '%s/numpy_files/figure-data-%s-%s-%s-seed%s.npy' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)

    log_path = '%s/log-%s-%s-%s-seed%s.txt' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)

    os.makedirs(os.path.dirname(save_path_final_npy), exist_ok=True)
    if os.path.exists(save_path_final_npy):
        data_numpy = np.load(save_path_final_npy)
        data_arrays = {
            'epoch': data_numpy['epoch'],
            'acc': data_numpy['acc'],
            'se': data_numpy['se'],
            'vne': data_numpy['vne'],
            'vne_random': data_numpy['vne_random_list'],
            'mi': data_numpy['mi'],
            'mi_sample': data_numpy['mi_sample']
            # 'mi_X': data_numpy['mi_X'],
            # 'mi_X_spectral': data_numpy['mi_X_spectral'],
        }
        plot_figures(data_arrays=data_arrays, save_paths_fig=save_paths_fig)

    else:
        epoch_list, acc_list, se_list, vne_list, vne_random_list, mi_Y_list, mi_Y_list_sample = [], [], [], [], [], [], []

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

            labels = labels.astype(np.int64)

            # if config.dataset == 'cifar10':
            #     labels_updated = np.zeros(labels.shape, dtype='object')
            #     for k in range(N):
            #         labels_updated[k] = cifar10_int2name[labels[k].item()]
            #     labels = labels_updated
            #     del labels_updated

            # diffusion_matrix = compute_diffusion_matrix(embeddings, k=args.knn)
            # print('ready set go')
            # import time
            # t1 = time.time()
            # eigenvalues_P1 = np.linalg.eigvals(diffusion_matrix)
            # t2 = time.time()
            # eigenvalues_P2 = approx_eigvals(diffusion_matrix)
            # t3 = time.time()
            # print(t2 - t1, t3 - t2)

            # import seaborn as sns
            # fig = plt.figure(figsize=(10, 10))
            # ax = fig.add_subplot(2, 2, 1)
            # sns.boxplot(x=eigenvalues_P1, color='skyblue', ax=ax)
            # ax.set_xlim([-1, 1])
            # ax = fig.add_subplot(2, 2, 2)
            # sns.boxplot(x=eigenvalues_P2, color='skyblue', ax=ax)
            # ax.set_xlim([-1, 1])
            # ax = fig.add_subplot(2, 2, 3)
            # ax.hist(eigenvalues_P1, color='white', edgecolor='k', bins=1000)
            # ax.set_xlim([-1, 1])
            # ax = fig.add_subplot(2, 2, 4)
            # ax.hist(eigenvalues_P2, color='white', edgecolor='k', bins=1000)
            # ax.set_xlim([-1, 1])
            # fig.savefig('test.png')

            # import pdb
            # pdb.set_trace()

            #
            '''Shannon Entropy of embeddings'''
            se = shannon_entropy(embeddings)
            se_list.append(se)
            log('Shannon Entropy = %.4f' % se, log_path)

            #
            '''Diffusion Matrix and Diffusion Eigenvalues & Vectors'''
            save_path_eigenvalues = '%s/numpy_files/diffusion-eigenvalues/diffusion-eigenvalues-fourier-coeff-%s.npz' % (
                save_root, checkpoint_name)
            os.makedirs(os.path.dirname(save_path_eigenvalues), exist_ok=True)
            if os.path.exists(save_path_eigenvalues):
                data_numpy = np.load(save_path_eigenvalues)
                eigenvalues_P = data_numpy['eigenvalues_P']
                coeffs_map = data_numpy['coeffs_map'] # [1 + (1+num_rep) x C, N]
                print('Pre-computed eigenvalues & fourier coeffs loaded.')
            else:
                diffusion_matrix = compute_diffusion_matrix(embeddings, sigma=args.gaussian_kernel_sigma)
                print('Diffusion matrix computed.')

                eigenvectors_P, eigenvalues_P = exact_eig(diffusion_matrix)

                '''Fourier Coeffs'''
                coeffs_map = []
                num_classes = int(np.max(labels) + 1)
                
                # First one is the full nodes coeffs
                signal = np.ones(N)
                coeffs = np.reshape(eigenvectors_P.T @ signal, (1,N))
                coeffs_map.append(coeffs)

                # Per Class coeffs
                for class_idx in tqdm(np.arange(num_classes)):
                    signal = np.zeros(N)
                    signal[labels==class_idx] = 1.0
                    coeffs = np.reshape(eigenvectors_P.T @ np.reshape(signal, (N, 1)), (1,N)) # N x 1
                    coeffs_map.append(coeffs)

                    # Random signals coeffs
                    for _ in np.arange(args.num_repetitions):
                        rand_inds = np.array(
                            random.sample(range(labels.shape[0]),
                                        k=np.sum(labels == class_idx)))
                        signal = np.zeros(N)
                        signal[labels==rand_inds] = 1.0
                        coeffs = np.reshape(eigenvectors_P.T @ np.reshape(signal, (N, 1)), (1,N))
                        coeffs_map.append(coeffs)
                        
                coeffs_map = np.vstack(coeffs_map)

                # Lower precision to save disk space.
                eigenvalues_P = eigenvalues_P.astype(np.float16)
                coeffs_map = coeffs_map.astype(np.float32)
                with open(save_path_eigenvalues, 'wb+') as f:
                    np.savez(f, eigenvalues_P=eigenvalues_P, coeffs_map=coeffs_map)
                print('Eigenvalues & Fourier Coeffs computed.')

            #
            '''Diffusion Entropy'''
            log('fourier_entropy Entropy: ', log_path)
            full_coeffs = coeffs_map[0, :] # First Row
            vne = fourier_entropy(full_coeffs, args.topk)
            vne_list.append(vne)
            log('Fourier Diffusion Entropy = %.4f' % vne, log_path)

            #
            '''Mutual Information between z and Output Class'''
            log('Mutual Information between z and Output Class: ', log_path)
            #classes_list, classes_cnts = np.unique(labels, return_counts=True)
            # vne_by_classes = []
            # for class_idx in tqdm(classes_list):
            #     inds = (labels == class_idx).reshape(-1)
            #     samples = embeddings[inds, :]
            #     # Diffusion Matrix
            #     diffusion_matrix_curr_class = compute_diffusion_matrix(
            #         samples, k=args.knn)
            #     # Eigenvalues
            #     eigenvalues_P_curr_class = np.linalg.eigvals(
            #         diffusion_matrix_curr_class)
            #     # Von Neumann Entropy
            #     vne_curr_class = von_neumann_entropy(eigenvalues_P_curr_class)
            #     vne_by_classes.append(vne_curr_class)
            # mi = mutual_information_per_class(eigenvalues_P,
            #                                   vne_by_classes,
            #                                   classes_cnts.tolist(),
            #                                   unconditioned_entropy=vne)
            
            mi_sample, H_ZgivenY = mi_fourier(coeffs_map[1:, :], args.num_repetitions)
            mi_full = vne - H_ZgivenY
            mi_Y_list.append(mi_full)
            log('MI between z and Output using full signal= %.4f' % mi_full, log_path)
            mi_Y_list_sample.append(mi_sample)
            log('MI between z and Output using random signal = %.4f' % mi_sample, log_path)

            vne_random = mi_sample + H_ZgivenY
            vne_random_list.append(vne_random)
            log('Fourier Diffusion Entropy Using random signals = %.4f' % vne_random, log_path)
            
            #
            '''Mutual Information between z and Input'''
            # log('Mutual Information between z and Input: ', log_path)
            # orig_input = np.reshape(orig_input,
            #                         (N, -1))  # [N, W, H, C] -> [N, W*H*C]
            # # MI with input H(z) - H(z|input)
            # mi_X, mi_cond, cond_classes_nums = mutual_information(
            #     orig_x=embeddings,
            #     cond_x=orig_input,
            #     knn=args.knn,
            #     class_method='bin',
            #     num_digit=2,
            #     orig_entropy=vne)

            # mi_X_list.append(mi_X)
            # log(
            #     'MI between z and Input = %.4f, Cond Entropy = %.4f, Cond Classes Num: %d '
            #     % (mi_X, mi_cond, cond_classes_nums), log_path)
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

            # mi_X_spectral, mi_cond_spectral, cond_classes_nums_spectral = mutual_information(
            #     orig_x=embeddings,
            #     cond_x=orig_input,
            #     knn=args.knn,
            #     class_method='spectral_bin',
            #     num_digit=2,
            #     num_spectral=None,
            #     diff_embed=diff_embed,
            #     orig_entropy=vne)

            # mi_X_spectral_list.append(mi_X_spectral)
            # log(
            #     '(Spectral Bin) MI between z and Input = %.4f, Cond Entropy = %.4f, Cond Classes Num: %d '
            #     % (mi_X_spectral, mi_cond_spectral, cond_classes_nums_spectral),
            #     log_path)

            # Plotting
            data_arrays = {
                'epoch': epoch_list,
                'acc': acc_list,
                'se': se_list,
                'vne': vne_list,
                'vne_random': vne_random_list,
                'mi': mi_Y_list,
                'mi_sample': mi_Y_list_sample,
                # 'mi_X': mi_X_list,
                # 'mi_X_spectral': mi_X_spectral_list,
            }
            plot_figures(data_arrays=data_arrays,
                         save_paths_fig=save_paths_fig)

        with open(save_path_final_npy, 'wb+') as f:
            np.savez(f,
                     epoch=np.array(epoch_list),
                     acc=np.array(acc_list),
                     se=np.array(se_list),
                     vne=np.array(vne_list),
                     vne_random=np.array(vne_random_list),
                     mi=np.array(mi_Y_list),
                     mi_sample=np.array(mi_Y_list_sample))
            #  mi_X=np.array(mi_X_list)
            #  mi_X_spectral=np.array(mi_X_spectral_list))

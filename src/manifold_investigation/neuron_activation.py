import argparse
import os
import random
import sys
from glob import glob
from typing import List

import numpy as np
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


def compute_act_stats(sampled_embeddings: np.array,
                      log_path: str,
                      log_str: str,
                      binary: bool = False) -> List[float]:
    if binary:
        correlation_matrix = np.corrcoef(sampled_embeddings > 0, rowvar=False)
    else:
        correlation_matrix = np.corrcoef(sampled_embeddings, rowvar=False)

    correlations = np.array([
        correlation_matrix[i, j] for i in range(len(correlation_matrix) - 1)
        for j in range(i + 1, len(correlation_matrix))
    ])
    # Correlation might have occasional NaN or inf values.
    correlations = np.array([
        item for item in correlations
        if (not np.isnan(item) and not np.isinf(item))
    ])

    if len(correlations) == 0:
        _mean = 1
        _std = 1
        _median = 1
        _25pctl = 1
        _75pctl = 1
    else:
        _mean = np.mean(correlations)
        _std = np.std(correlations)
        _median = np.median(correlations)
        _25pctl = np.percentile(correlations, 25)
        _75pctl = np.percentile(correlations, 75)

    log('%s:' % log_str, log_path)
    log('    Mean \u00B1 std: %.3f \u00B1 %.3f' % (_mean, _std), log_path)
    log('    Median: %.3f' % _median, log_path)

    return _mean, _std, _25pctl, _25pctl, _75pctl


def plot_helper(ax: plt.Axes, epoch_list: List[float], acc_list: List[float],
                _mean_list: List[float], _std_list: List[float],
                _median_list: List[float], _25pctl_list: List[float],
                _75pctl_list: List[float], ylabel_str: str) -> None:

    ax.spines[['right', 'top']].set_visible(False)

    # Left Axis: Activation statistics.
    ax.plot(epoch_list, _mean_list, c='mediumblue', lw=5)
    ax.plot(epoch_list, _median_list, c='k', lw=5)
    ax.legend(['mean \u00B1 std', 'median \u00B1 25 percentiles'],
              fontsize=30,
              loc='lower right')
    ax.fill_between(epoch_list,
                    np.array(_mean_list) - np.array(_std_list),
                    np.array(_mean_list) + np.array(_std_list),
                    color='mediumblue',
                    alpha=0.2)
    ax.fill_between(epoch_list,
                    _25pctl_list,
                    _75pctl_list,
                    color='k',
                    alpha=0.2)

    ax.set_xlabel('Epochs Trained', fontsize=40)
    ax.set_ylabel(ylabel_str, fontsize=38)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Right axis: Accuracy
    ax_secondary = ax.twinx()
    ax_secondary.plot(epoch_list, acc_list, c='firebrick', lw=5)
    ax_secondary.yaxis.set_label_coords(1.1, 0.5)
    ax_secondary.set_ylabel('Downstream Classification Accuracy',
                            fontsize=38,
                            rotation=270)
    ax_secondary.tick_params(axis='both', which='major', labelsize=30)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument(
        '--random-seed',
        help='Only enter if you want to override the config!!!',
        type=int,
        default=None)
    parser.add_argument('--num-observation-samples', type=int, default=500)

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

    save_root = './results_neuron_activation/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig_act_corr = '%s/neuron-activation-corr-%s-%s-%s-seed%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)
    save_path_fig_act_bin_corr = '%s/neuron-activation-bin-corr-%s-%s-%s-seed%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)
    log_path = '%s/log-%s-%s-%s-seed%s.txt' % (save_root, config.dataset,
                                               method_str, config.model,
                                               config.random_seed)

    random.seed(config.random_seed)

    save_path_act_stats = '%s/numpy_files/activation-stats-%s-%s-%s-seed%s.npz' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)
    os.makedirs(os.path.dirname(save_path_act_stats), exist_ok=True)

    act_description = 'Final Layer Neuron Activation (Output) Correlation'
    act_bin_description = 'Final Layer Neuron Activation (On/Off) Correlation'

    if os.path.exists(save_path_act_stats):
        data_numpy = np.load(save_path_act_stats)
        epoch_list = data_numpy['epoch_list']
        acc_list = data_numpy['acc_list']
        act_corr_mean_list = data_numpy['act_corr_mean_list']
        act_corr_std_list = data_numpy['act_corr_std_list']
        act_corr_median_list = data_numpy['act_corr_median_list']
        act_corr_25pctl_list = data_numpy['act_corr_25pctl_list']
        act_corr_75pctl_list = data_numpy['act_corr_75pctl_list']
        act_bin_corr_mean_list = data_numpy['act_bin_corr_mean_list']
        act_bin_corr_std_list = data_numpy['act_bin_corr_std_list']
        act_bin_corr_median_list = data_numpy['act_bin_corr_median_list']
        act_bin_corr_25pctl_list = data_numpy['act_bin_corr_25pctl_list']
        act_bin_corr_75pctl_list = data_numpy['act_bin_corr_75pctl_list']
        print('Pre-computed activation stats loaded.')

    else:
        epoch_list, acc_list = [], []
        act_corr_mean_list, act_corr_std_list = [], []
        act_corr_median_list, act_corr_25pctl_list, act_corr_75pctl_list = [], [], []
        act_bin_corr_mean_list, act_bin_corr_std_list = [], []
        act_bin_corr_median_list, act_bin_corr_25pctl_list, act_bin_corr_75pctl_list = [], [], []

        for embedding_folder in embedding_folders:
            epoch_list.append(
                int(embedding_folder.split('epoch')[-1].split('-valAcc')[0]) +
                1)
            acc_list.append(float(embedding_folder.split('-valAcc')[1]))

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

            # NOTE: This time we only consider the "activation" of neurons in the
            # last layer before the final fully-connected classifier.
            if args.num_observation_samples is not None:
                obs_ids = random.choices(np.arange(N),
                                         k=args.num_observation_samples)
                sampled_embeddings = embeddings[obs_ids, :]
            else:
                sampled_embeddings = embeddings

            _mean, _std, _median, _25pctl, _75pctl = compute_act_stats(
                sampled_embeddings=sampled_embeddings,
                log_path=log_path,
                log_str=act_description,
                binary=False)

            act_corr_mean_list.append(_mean)
            act_corr_std_list.append(_std)
            act_corr_median_list.append(_median)
            act_corr_25pctl_list.append(_25pctl)
            act_corr_75pctl_list.append(_75pctl)

            _mean, _std, _median, _25pctl, _75pctl = compute_act_stats(
                sampled_embeddings=sampled_embeddings,
                log_path=log_path,
                log_str=act_bin_description,
                binary=True)

            act_bin_corr_mean_list.append(_mean)
            act_bin_corr_std_list.append(_std)
            act_bin_corr_median_list.append(_median)
            act_bin_corr_25pctl_list.append(_25pctl)
            act_bin_corr_75pctl_list.append(_75pctl)

        with open(save_path_act_stats, 'wb+') as f:
            np.savez(
                f,
                epoch_list=np.array(epoch_list),
                acc_list=np.array(acc_list),
                act_corr_mean_list=np.array(act_corr_mean_list),
                act_corr_std_list=np.array(act_corr_std_list),
                act_corr_median_list=np.array(act_corr_median_list),
                act_corr_25pctl_list=np.array(act_corr_25pctl_list),
                act_corr_75pctl_list=np.array(act_corr_75pctl_list),
                act_bin_corr_mean_list=np.array(act_bin_corr_mean_list),
                act_bin_corr_std_list=np.array(act_bin_corr_std_list),
                act_bin_corr_median_list=np.array(act_bin_corr_median_list),
                act_bin_corr_25pctl_list=np.array(act_bin_corr_25pctl_list),
                act_bin_corr_75pctl_list=np.array(act_bin_corr_75pctl_list))

    #
    ''' Plotting '''
    plt.rcParams['font.family'] = 'serif'

    subplot_params = {
        'left': 0.05,
        'right': 0.95,
        'bottom': 0.1,
        'top': 0.9,
        'wspace': 0.3,
        'hspace': 0.3
    }

    # Find where accuracy plateaus.
    plateau_acc = np.max(acc_list) * 0.98
    plateau_idx = np.argwhere(np.array(acc_list) > plateau_acc)[0][0]

    #
    ''' Figure for Neuron Activation (Output) '''
    fig_act_corr = plt.figure(figsize=(48, 20))

    # Plot entire history.
    plot_helper(ax=fig_act_corr.add_subplot(1, 2, 1),
                epoch_list=epoch_list,
                acc_list=acc_list,
                _mean_list=act_corr_mean_list,
                _std_list=act_corr_std_list,
                _median_list=act_corr_median_list,
                _25pctl_list=act_corr_25pctl_list,
                _75pctl_list=act_corr_75pctl_list,
                ylabel_str=act_description)
    # Plot up to where accuracy plateaus.
    plot_helper(ax=fig_act_corr.add_subplot(1, 2, 2),
                epoch_list=epoch_list[:plateau_idx],
                acc_list=acc_list[:plateau_idx],
                _mean_list=act_corr_mean_list[:plateau_idx],
                _std_list=act_corr_std_list[:plateau_idx],
                _median_list=act_corr_median_list[:plateau_idx],
                _25pctl_list=act_corr_25pctl_list[:plateau_idx],
                _75pctl_list=act_corr_75pctl_list[:plateau_idx],
                ylabel_str=act_description)

    fig_act_corr.subplots_adjust(**subplot_params)
    fig_act_corr.savefig(save_path_fig_act_corr)
    plt.close(fig=fig_act_corr)

    #
    ''' Figure for Binary Activation (On/Off) '''
    fig_act_bin_corr = plt.figure(figsize=(48, 20))
    # Plot entire history.
    plot_helper(ax=fig_act_bin_corr.add_subplot(1, 2, 1),
                epoch_list=epoch_list,
                acc_list=acc_list,
                _mean_list=act_bin_corr_mean_list,
                _std_list=act_bin_corr_std_list,
                _median_list=act_bin_corr_median_list,
                _25pctl_list=act_bin_corr_25pctl_list,
                _75pctl_list=act_bin_corr_75pctl_list,
                ylabel_str=act_bin_description)
    # Plot up to where accuracy plateaus.
    plot_helper(ax=fig_act_bin_corr.add_subplot(1, 2, 2),
                epoch_list=epoch_list[:plateau_idx],
                acc_list=acc_list[:plateau_idx],
                _mean_list=act_bin_corr_mean_list[:plateau_idx],
                _std_list=act_bin_corr_std_list[:plateau_idx],
                _median_list=act_bin_corr_median_list[:plateau_idx],
                _25pctl_list=act_bin_corr_25pctl_list[:plateau_idx],
                _75pctl_list=act_bin_corr_75pctl_list[:plateau_idx],
                ylabel_str=act_bin_description)

    fig_act_bin_corr.subplots_adjust(**subplot_params)
    fig_act_bin_corr.savefig(save_path_fig_act_bin_corr)
    plt.close(fig=fig_act_bin_corr)

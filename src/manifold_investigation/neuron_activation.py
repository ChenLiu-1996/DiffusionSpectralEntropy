import argparse
import os
import sys
from glob import glob
from typing import List

import numpy as np
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import random
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
    parser.add_argument('--sampled-observations', type=int, default=None)

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
    log_path = '%s/log-%s-%s-%s-seed%s.txt' % (save_root, config.dataset,
                                               method_str, config.model,
                                               config.random_seed)

    num_rows = len(embedding_folders)
    epoch_list, acc_list = [], []
    act_corr_mean_list, act_corr_std_list = [], []
    act_corr_median_list, act_corr_25pctl_list, act_corr_75_pctl_list = [], [], []

    random.seed(config.random_seed)

    for i, embedding_folder in enumerate(embedding_folders):

        epoch_list.append(
            int(embedding_folder.split('epoch')[-1].split('-valAcc')[0]) + 1)
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
        #
        # Random downsampling of observations
        obs_ids = random.choices(np.arange(N),
                                 k=500 if args.sampled_observations is None
                                 else args.sampled_observations)
        sampled_embeddings = embeddings[obs_ids, :]

        correlation_matrix = np.corrcoef(sampled_embeddings,
                                         rowvar=False,
                                         dtype=np.float16)
        correlations = np.array([
            correlation_matrix[i, j]
            for i in range(len(correlation_matrix) - 1)
            for j in range(i + 1, len(correlation_matrix))
        ],
                                dtype=np.float32)
        # Correlation might have occasional NaN or inf values.
        correlations = np.array([
            item for item in correlations
            if (not np.isnan(item) and not np.isinf(item))
        ])
        act_corr_mean = np.mean(correlations)
        act_corr_std = np.std(correlations)
        act_corr_median = np.median(correlations)
        act_corr_25pctl = np.percentile(correlations, 25)
        act_corr_75pctl = np.percentile(correlations, 75)

        log('Finaly layer neuron activation (Output) correlation stats:',
            log_path)
        log(
            '    Mean \u00B1 std: %.3f \u00B1 %.3f' %
            (act_corr_mean, act_corr_std), log_path)
        log('    Median: %.3f' % act_corr_median, log_path)

        act_corr_mean_list.append(act_corr_mean)
        act_corr_std_list.append(act_corr_std)
        act_corr_median_list.append(act_corr_median)
        act_corr_25pctl_list.append(act_corr_25pctl)
        act_corr_75_pctl_list.append(act_corr_75pctl)

        #
        '''Plotting'''
        plt.rcParams['font.family'] = 'serif'
        fig_act_corr = plt.figure(figsize=(20, 20))
        ax = fig_act_corr.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.plot(epoch_list, act_corr_mean_list, c='mediumblue')
        ax.plot(epoch_list, act_corr_median_list, c='k')
        ax.legend(['mean \u00B1 std', 'median \u00B1 25 percentiles'],
                  fontsize=30)
        ax.fill_between(
            epoch_list,
            np.array(act_corr_mean_list) - np.array(act_corr_std_list),
            np.array(act_corr_mean_list) + np.array(act_corr_std_list),
            color='mediumblue',
            alpha=0.2)
        ax.fill_between(epoch_list,
                        act_corr_25pctl_list,
                        act_corr_75_pctl_list,
                        color='k',
                        alpha=0.2)
        fig_act_corr.supylabel(
            'Final Encoder Layer Neuron Activation (Output) Correlation',
            fontsize=38)
        fig_act_corr.supxlabel('Epochs Trained', fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        fig_act_corr.savefig(save_path_fig_act_corr)
        plt.close(fig=fig_act_corr)

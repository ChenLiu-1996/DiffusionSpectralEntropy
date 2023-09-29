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
import torch
from scipy.stats import pearsonr, spearmanr

import yaml
from matplotlib import pyplot as plt

from tqdm import tqdm
from _timm_models import build_timm_model

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from attribute_hashmap import AttributeHashmap
from path_utils import update_config_dirs

train_embeddings_utils = __import__('01_train_embeddings')


def compute_embeddings(model, loader):
    embeddings, labels = None, None
    with torch.no_grad():
        for x, y_true in tqdm(loader):
            B = x.shape[0]
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)

            # TODO: Downsample the input image to reduce memory usage?
            curr_X = torch.nn.functional.interpolate(
                x, size=(64, 64)).cpu().numpy().reshape(x.shape[0], -1)
            curr_Y = y_true.cpu().numpy()
            curr_Z = model.encode(x).cpu().numpy()

            if labels is None:
                labels = curr_Y.reshape(B, 1)
                embeddings = curr_Z
            else:
                labels = np.vstack((labels, curr_Y.reshape(B, 1)))
                embeddings = np.vstack((embeddings, curr_Z))

    N, D = embeddings.shape

    assert labels.shape[0] == N
    assert labels.shape[1] == 1
    return embeddings, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument(
        '--model',
        help='model name: [resnet, resnext, convnext, vit, swin, xcit]',
        type=str,
        required=True)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument(
        '--random-seed',
        help='Only enter if you want to override the config!!!',
        type=int,
        default=None)
    parser.add_argument('--num-bins',
                        help='Number of bins for histogram',
                        type=int,
                        default=100)
    parser.add_argument(
        '--summary-bins',
        help='Number of bins for summary histogram, should be much smaller',
        type=int,
        default=5)
    parser.add_argument(
        '--conv-init-std',
        help='conv block initialization std (please use this format: 1e-3)',
        type=str,
        required=True)

    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.model = args.model
    config.num_bins = args.num_bins
    config.summary_bins = args.summary_bins

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    method_str = config.method
    save_root = './results_coupling/'
    os.makedirs(save_root, exist_ok=True)

    # Initialization Experiments, take epoch checkpoints
    checkpoints = sorted(
        glob('%s/%s-%s-%s-ConvInitStd-%s-seed%s/*.pth' %
             (config.checkpoint_dir, config.dataset, config.method,
              config.model, args.conv_init_std, config.random_seed)))
    save_path_fig = '%s/coupling-%s-%s-%s-ConvInitSdt-%s-seed%s' % (
        save_root, config.dataset, method_str, config.model,
        args.conv_init_std, config.random_seed)
    print('%s/%s-%s-%s-ConvInitStd-%s-seed%s/*.pth' %
          (config.checkpoint_dir, config.dataset, config.method, config.model,
           args.conv_init_std, config.random_seed))
    print('checkpoints: ', len(checkpoints))
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

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    seed_everything(0)

    plt.rcParams['font.family'] = 'serif'
    num_rows = len(checkpoints) + 1
    if num_rows > 20:
        num_rows = 20

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 16
    fig_pr = plt.figure(figsize=(15, 1 * num_rows))
    fig_sr = plt.figure(figsize=(15, 1 * num_rows))

    # Load model and run inference.
    dataloaders, config = train_embeddings_utils.get_dataloaders(config=config)
    _, val_loader = dataloaders

    model = build_timm_model(model_name=config.model,
                             num_classes=config.num_classes).to(device)

    for i in range(num_rows):
        if i == 0:
            model.init_params(conv_init_std=float(args.conv_init_std))
        else:
            checkpoint_name = checkpoints[i - 1]
            model.load_state_dict(
                torch.load(checkpoint_name, map_location=device))
        model.eval()

        embeddings, labels = compute_embeddings(model, val_loader)

        # Compute Pearson Correlation, Spearman Correlation between D x N neurons
        neurons = embeddings.T
        #NOTE: Pairwise Pearson R is sooooo slow!
        # pr = np.empty((neurons.shape[0], neurons.shape[0]))
        # for i in range(neurons.shape[0]):
        #     for j in range(i, neurons.shape[0]):
        #         value = pearsonr(neurons[i], neurons[j])[0]
        #         pr[i][j] = value
        #         pr[j][i] = value

        sr, _ = spearmanr(neurons, axis=1)

        # assert pr.shape[0] == neurons.shape[0]
        # assert pr.shape[1] == neurons.shape[0]
        assert sr.shape[0] == neurons.shape[0]
        assert sr.shape[1] == neurons.shape[0]

        # # Plot histogram
        # ax = fig_pr.add_subplot(num_rows, 1, i + 1)
        # ax.spines[['right', 'top', 'left']].set_visible(False)
        # ax.hist(sr.flatten(),
        #         bins=config.num_bins,
        #         color='white',
        #         edgecolor='black')
        # ax.set_xlim([-1, 1])
        # ax.set_yticks([])
        # ax.set_yticklabels([])

        # # Summary of histogram
        # title_str = '|R|>0.9: %.2f%%, |R|>0.8: %.2f%%, |R|>0.5: %.2f%%' % (
        #     (np.abs(sr) > 0.9).sum() / len(sr.flatten()) * 100,
        #     (np.abs(sr) > 0.8).sum() / len(sr.flatten()) * 100,
        #     (np.abs(sr) > 0.5).sum() / len(sr.flatten()) * 100)
        # ax.set_title(title_str)

        ax = fig_sr.add_subplot(num_rows, 1, i + 1)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.hist(sr.flatten(),
                bins=config.num_bins,
                color='skyblue',
                edgecolor='white',
                alpha=0.5)
        ax.set_xlim([-1, 1])
        ax.set_yticks([])
        ax.set_yticklabels([])

        title_str = '|R|>0.9: %.2f%%, |R|>0.8: %.2f%%, |R|>0.5: %.2f%%' % (
            (np.abs(sr) > 0.9).sum() / len(sr.flatten()) * 100,
            (np.abs(sr) > 0.8).sum() / len(sr.flatten()) * 100,
            (np.abs(sr) > 0.5).sum() / len(sr.flatten()) * 100)
        ax.set_title(title_str, fontsize=16)

        # fig_pr.tight_layout()
        # fig_pr.savefig(save_path_fig + 'PearsonR')
        fig_sr.tight_layout()
        fig_sr.savefig(save_path_fig + 'SpearmanR')

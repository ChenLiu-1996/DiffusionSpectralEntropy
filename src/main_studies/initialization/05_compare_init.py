import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import argparse
import sys
import seaborn as sns
import numpy as np
import torch

import yaml
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from attribute_hashmap import AttributeHashmap
from path_utils import update_config_dirs

train_embeddings_utils = __import__('01_train_embeddings')


def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height],
                         facecolor=facecolor)  # matplotlib 2.0+
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


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

    # NOTE: For simplicity of legend, I just manually set the std now.
    args.conv_init_std = ['1e-2', '1e-1']

    method_str = config.method
    save_root = './results_compare/'
    os.makedirs(save_root, exist_ok=True)

    # Initialization Experiments, take training history.
    training_history_dict = {}
    for conv_init_std in args.conv_init_std:
        save_path_numpy = '%s/%s-%s-%s-ConvInitStd-%s-seed%s/%s' % (
            config.output_save_path, config.dataset, config.method,
            config.model, conv_init_std, config.random_seed, 'results.npz')
        results_dict = np.load(save_path_numpy)
        training_history_dict[conv_init_std] = {
            'epoch': results_dict['epoch'],
            'val_acc': results_dict['val_acc'],
            'dse_Z': results_dict['dse_Z'],
        }

    save_path_fig = '%s/compare-%s-%s-%s-ConvInitStd-%s-seed%s' % (
        save_root, config.dataset, method_str, config.model, '-'.join(
            [item for item in args.conv_init_std]), config.random_seed)
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

    my_palette = sns.color_palette('icefire', n_colors=len(args.conv_init_std))
    plt.rcParams['font.family'] = 'serif'
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 3)

    ax = fig.add_subplot(gs[0])
    ax.spines[['right', 'top']].set_visible(False)
    for i, conv_init_std in enumerate(args.conv_init_std):
        ax.plot(
            training_history_dict[conv_init_std]['epoch'][:15],
            training_history_dict[conv_init_std]['dse_Z'][:15],
            color=my_palette[i],
            linestyle='-.',
        )
    ax.legend([r'conv. init. std. = 0.01', r'conv. init. std. = 0.1'])
    ax.set_xlabel('Epochs Trained')
    ax.set_ylabel(r'DSE $S_D(Z)$')

    ax = fig.add_subplot(gs[1:])
    ax.spines[['right', 'top']].set_visible(False)
    ax.add_patch(
        patches.Rectangle((50, 85),
                          152,
                          11,
                          linewidth=1,
                          linestyle='--',
                          edgecolor='darkgreen',
                          facecolor='none',
                          label='_nolegend_'))

    rect = [0.35, 0.35, 0.6, 0.56]
    ax_sub = add_subplot_axes(ax, rect)
    ax_sub.spines[['right', 'top']].set_visible(False)

    for i, conv_init_std in enumerate(args.conv_init_std):
        ax.plot(
            training_history_dict[conv_init_std]['epoch'],
            training_history_dict[conv_init_std]['val_acc'],
            color=my_palette[i],
        )
        ax_sub.plot(
            training_history_dict[conv_init_std]['epoch'][50:],
            training_history_dict[conv_init_std]['val_acc'][50:],
            color=my_palette[i],
            label='_nolegend_',
        )
    ax.legend([
        r'Initial DSE $S_D$(Z) is low (conv. init. std. = 0.01)',
        r'Initial DSE $S_D$(Z) is high (conv. init. std. = 0.1)'
    ])
    ax.set_xlabel('Epochs Trained')
    ax.set_ylabel('Val. Accuracy')

    fig.tight_layout()
    fig.savefig(save_path_fig)
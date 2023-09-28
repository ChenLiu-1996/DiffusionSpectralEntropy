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

import phate
import scprep
import yaml
from matplotlib import pyplot as plt

from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from seed import seed_everything
from attribute_hashmap import AttributeHashmap
from path_utils import update_config_dirs
from timm_models import build_timm_model

train_embeddings_utils = __import__('01_train_embeddings')

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
    parser.add_argument(
        '--num-bins',
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

    if args.random_seed is not None:
        config.random_seed = args.random_seed
    method_str = config.method
    # NOTE: Take the fixed percentage checkpoints.
    checkpoints = sorted(
        glob('%s/%s-%s-%s-seed%s/*.pth' %
             (config.checkpoint_dir, config.dataset, config.method,
              config.model, config.random_seed)))

    save_root = './results_coupling/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig = '%s/coupling-%s-%s-%s-seed%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)

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
    num_rows = len(checkpoints)
    fig = plt.figure(figsize=(9, 3 * num_rows))

    for i, checkpoint_name in enumerate(checkpoints):
        if config.method == 'wronglabel':
            val_metric = 'acc_diverg'
        else:
            val_metric = 'val_acc'

        val_metric_pc = checkpoint_name.split(val_metric +
                                              '_')[1].split('.')[0]

        # Load model and run inference.
        dataloaders, config = train_embeddings_utils.get_dataloaders(
            config=config)
        _, val_loader = dataloaders

        model = build_timm_model(model_name=config.model,
                                 num_classes=config.num_classes).to(device)
        model.init_params()
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
        model.eval()

        labels, embeddings = None, None
        with torch.no_grad():
            for x, y_true in tqdm(val_loader):
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


        # Compute Pearson Correlation, Spearman Correlation between D x N neurons
        neurons = embeddings.T
        sr, _ = spearmanr(neurons, axis=1)

        assert sr.shape[0] == D
        assert sr.shape[1] == D

        # Plot histogram
        ax = fig.add_subplot(num_rows, 1, i + 1)
        ax.spines[['right', 'top', 'left']].set_visible(False)
        ax.hist(sr.flatten(), bins=config.num_bins)

        # Summary of histogram
        counts, bins = np.histogram(sr, bins=config.summary_bins)
        title_str = ""
        for i in range(len(bins)) - 1:
            title_str += '%d - %d: %d'%(bins[i], bins[i+1], counts[i])
            title_str += "\n"
        ax.set_title(title_str)


    fig.tight_layout()
    fig.savefig(save_path_fig)




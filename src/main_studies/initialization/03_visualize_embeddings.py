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
import phate
import scprep

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
    save_root = './results_visualize/'
    os.makedirs(save_root, exist_ok=True)

    # Initialization Experiments, take epoch checkpoints
    checkpoints = sorted(
        glob('%s/%s-%s-%s-ConvInitStd-%s-seed%s/*.pth' %
             (config.checkpoint_dir, config.dataset, config.method,
              config.model, args.conv_init_std, config.random_seed)))
    # Initialization Experiments, take training history.
    save_path_numpy = '%s/%s-%s-%s-ConvInitStd-%s-seed%s/%s' % (
        config.output_save_path, config.dataset, config.method, config.model,
        config.conv_init_std, config.random_seed, 'results.npz')

    save_path_fig = '%s/phate-%s-%s-%s-ConvInitStd-%s-seed%s' % (
        save_root, config.dataset, method_str, config.model,
        args.conv_init_std, config.random_seed)

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

    results_dict = np.load(save_path_numpy)
    dse_Z = results_dict['dse_Z']

    plt.rcParams['font.family'] = 'serif'
    epochs = [0, 1, 2, 3, 4, 10, 20, 50, 100, 200]
    num_cols = len(epochs)
    fig = plt.figure(figsize=(4 * num_cols, 5))

    # Load model and run inference.
    dataloaders, config = train_embeddings_utils.get_dataloaders(config=config)
    _, val_loader = dataloaders

    model = build_timm_model(model_name=config.model,
                             num_classes=config.num_classes).to(device)

    for i in range(num_cols):
        if i == 0:
            model.init_params(conv_init_std=float(args.conv_init_std))
        else:
            epoch_idx = epochs[i]
            checkpoint_name = checkpoints[epoch_idx - 1]
            model.load_state_dict(
                torch.load(checkpoint_name, map_location=device))
        model.eval()

        embeddings, labels = compute_embeddings(model, val_loader)

        #
        ''' Plotting '''
        #
        ''' PHATE plot, colored by class. '''
        ax = fig.add_subplot(1, num_cols, i + 1)
        phate_op = phate.PHATE(random_state=0,
                               n_jobs=1,
                               n_components=2,
                               t=10,
                               verbose=False)
        data_phate = phate_op.fit_transform(embeddings)
        scprep.plot.scatter2d(data_phate,
                              c=labels,
                              legend=False,
                              ax=ax,
                              title='epoch %s, DSE(Z) = %.1f' %
                              (epochs[i], dse_Z[epochs[i] - 1]),
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=16,
                              s=3)

        fig.tight_layout()
        fig.savefig(save_path_fig)
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

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from laplacian_extrema import get_laplacian_extrema
from path_utils import update_config_dirs
from timm_models import build_timm_model

# sys.path.insert(0, import_dir + '/src/main_studies/training_dynamic/')
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
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.model = args.model
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    method_str = config.method
    
    # NOTE: Take the fixed percentage checkpoints.
    checkpoints = sorted(
        glob('%s/%s-%s-%s-seed%s-*.pth' % (
            config.checkpoint_dir, config.dataset,
            config.method, config.model, config.random_seed))
    )

    save_root = './results_visualize/'
    os.makedirs(save_root, exist_ok=True)
    save_path_fig = '%s/visualize-%s-%s-%s-seed%s.png' % (
        save_root, config.dataset, method_str, config.model,
        config.random_seed)

    plt.rcParams['font.family'] = 'serif'
    num_cols = len(checkpoints)
    fig = plt.figure(figsize=(3 * num_cols, 6))

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
    
    for i, checkpoint_name in enumerate(checkpoints):
        if config.method == 'wronglabel':
            val_metric = 'acc_diverg'
        else:
            val_metric = 'val_acc'        
        
        val_metric_pc = checkpoint_name.split(val_metric + '_')[1].split('.')[0]

        # Load model and run inference.
        dataloaders, config = train_embeddings_utils.get_dataloaders(config=config)
        _, val_loader, _ = dataloaders

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

        #
        ''' Laplacian extrema '''
        save_path_extrema = '%s/numpy_files/laplacian-extrema/laplacian-extrema-%s.npz' % (
            save_root, checkpoint_name)
        os.makedirs(os.path.dirname(save_path_extrema), exist_ok=True)
        if os.path.exists(save_path_extrema):
            data_numpy = np.load(save_path_extrema)
            extrema_inds = data_numpy['extrema_inds']
            print('Pre-computed Laplacian extrema loaded.')
        else:
            n_extrema = num_classes
            extrema_inds = get_laplacian_extrema(data=embeddings,
                                                 n_extrema=n_extrema,
                                                 knn=args.knn)
            with open(save_path_extrema, 'wb+') as f:
                np.savez(f, extrema_inds=extrema_inds)
            print('Laplacian extrema computed.')

        #
        ''' Plotting '''
        #
        ''' PHATE plot, colored by class. '''
        ax = fig.add_subplot(2, num_cols, i + 1)
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
                              title=val_metric_pc,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=3)

        #
        ''' PHATE plot, annotated by Laplacian extrema. '''
        ax = fig.add_subplot(2, num_cols, i + 1 + num_cols)

        colors_extrema = np.empty((N), dtype=object)
        colors_extrema.fill('Embedding\nVectors')
        colors_extrema[extrema_inds] = 'Laplacian\nExtrema'
        cmap_extrema = {
            'Embedding\nVectors': 'gray',
            'Laplacian\nExtrema': 'firebrick'
        }
        sizes = np.empty((N), dtype=int)
        sizes.fill(1)
        sizes[extrema_inds] = 50

        scprep.plot.scatter2d(data_phate,
                              c=colors_extrema,
                              cmap=cmap_extrema,
                              title=None,
                              legend=False,
                              ax=ax,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=sizes)

        fig.tight_layout()
        fig.savefig(save_path_fig)

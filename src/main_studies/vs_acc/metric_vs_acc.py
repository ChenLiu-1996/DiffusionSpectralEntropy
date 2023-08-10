import argparse
import os
import sys
from typing import Tuple, Dict, Iterable
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import pandas as pd

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm
import timm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy
from dsmi import diffusion_spectral_mutual_information

sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from seed import seed_everything


def get_val_loader(
    args: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
    if args.dataset == 'mnist':
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif args.dataset == 'cifar10':
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif args.dataset == 'stl10':
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    elif args.dataset == 'tinyimagenet':
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet

    elif args.dataset == 'imagenet':
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet

    else:
        raise ValueError(
            '`args.dataset` value not supported. Value provided: %s.' %
            args.dataset)

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            args.imsize,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(args.imsize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_val)

    elif args.dataset in ['stanfordcars', 'stl10', 'food101', 'flowers102']:
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          split='test',
                                          download=True,
                                          transform=transform_val)

    elif args.dataset in ['tinyimagenet', 'imagenet']:
        val_dataset = torchvision_dataset(args.dataset_dir,
                                          split='val',
                                          transform=transform_val)

    # shuffle=True because we want to only sample ~5k data points
    # for efficient DSE/DSMI computation, but meanwhile want to
    # maintain diversity of labels.
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=True,
                                             pin_memory=True)
    return val_loader


class ThisArchitectureIsWeirdError(Exception):
    pass


def plot_figures(data_arrays: Dict[str, Iterable], save_path_fig: str) -> None:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    # Plot of DSE vs val top 1 acc.
    fig = plt.figure(figsize=(40, 20))
    img_idx = 1
    for y_str in [
            'imagenet_val_acc_top1', 'imagenet_val_acc_top5',
            'imagenet_test_acc_top1', 'imagenet_test_acc_top5'
    ]:
        for x_str in [
                'dse_Z', 'dsmi_Z_X', 'dsmi_Z_Y', 'cse_Z', 'csmi_Z_X',
                'csmi_Z_Y'
        ]:
            ax = fig.add_subplot(4, 6, img_idx)
            img_idx += 1
            plot_subplot(ax, data_arrays, x_str, y_str)
    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig=fig)
    return


def plot_subplot(ax: plt.Axes, data_arrays: Dict[str, Iterable], x_str: str,
                 y_str: str) -> plt.Axes:
    arr_title_map = {
        'imagenet_val_acc_top1': 'ImageNet val top1-acc',
        'imagenet_val_acc_top5': 'ImageNet val top5-acc',
        'imagenet_test_acc_top1': 'ImageNet test top1-acc',
        'imagenet_test_acc_top5': 'ImageNet test top5-acc',
        'dse_Z': 'DSE(Z)',
        'cse_Z': 'CSE(Z)',
        'dsmi_Z_X': 'DSMI(Z; X)',
        'csmi_Z_X': 'CSMI(Z; X)',
        'dsmi_Z_Y': 'DSMI(Z; Y)',
        'csmi_Z_Y': 'CSMI(Z; Y)',
    }
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays[x_str],
               data_arrays[y_str],
               c='darkblue',
               alpha=0.2,
               s=np.array(data_arrays['model_params']) / 4)
    ax.set_xlabel(arr_title_map[x_str], fontsize=20)
    ax.set_ylabel(arr_title_map[y_str], fontsize=20)
    if len(data_arrays[x_str]) > 1:
        ax.set_title('P.R: %.3f (p = %.3f), S.R: %.3f (p = %.3f)' % \
                    (pearsonr(data_arrays[x_str], data_arrays[y_str])[0],
                    pearsonr(data_arrays[x_str], data_arrays[y_str])[1],
                    spearmanr(data_arrays[x_str], data_arrays[y_str])[0],
                    spearmanr(data_arrays[x_str], data_arrays[y_str])[1]),
                    fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=20)
    return


class ModelWithLatentAccess(torch.nn.Module):

    def __init__(self,
                 timm_model: torch.nn.Module,
                 num_classes: int = 10) -> None:
        super(ModelWithLatentAccess, self).__init__()
        self.num_classes = num_classes

        # Isolate the model into an encoder and a linear classifier.
        self.encoder = timm_model

        # Get the correct dimensions of the last linear layer and remove the linear layer.
        # The last linear layer may come with different names...
        if any(n == 'fc' for (n, _) in self.encoder.named_children()) and \
           isinstance(self.encoder.fc, torch.nn.modules.Linear):
            last_layer = self.encoder.fc
            last_layer_name_opt = 1
        elif any(n == 'classifier' for (n, _) in self.encoder.named_children()) and \
           isinstance(self.encoder.classifier, torch.nn.modules.Linear):
            last_layer = self.encoder.classifier
            last_layer_name_opt = 2
        elif any(n == 'head' for (n, _) in self.encoder.named_children()) and \
           isinstance(self.encoder.head, torch.nn.modules.Linear):
            last_layer = self.encoder.head
            last_layer_name_opt = 3
        elif any(n == 'head' for (n, _) in self.encoder.named_children()) and \
             any(n == 'fc' for (n, _) in self.encoder.head.named_children()) and \
           isinstance(self.encoder.head.fc, torch.nn.modules.Linear):
            last_layer = self.encoder.head.fc
            last_layer_name_opt = 4
        elif any(n == 'head' for (n, _) in self.encoder.named_children()) and \
             any(n == 'fc' for (n, _) in self.encoder.head.named_children()) and \
             any(n == 'fc2' for (n, _) in self.encoder.head.fc.named_children()) and \
           isinstance(self.encoder.head.fc.fc2, torch.nn.modules.Linear):
            last_layer = self.encoder.head.fc.fc2
            last_layer_name_opt = 5
        else:
            raise ThisArchitectureIsWeirdError

        assert last_layer.out_features == num_classes

        self.linear = last_layer

        if last_layer_name_opt == 1:
            self.encoder.fc = torch.nn.Identity()
        elif last_layer_name_opt == 2:
            self.encoder.classifier = torch.nn.Identity()
        elif last_layer_name_opt == 3:
            self.encoder.head = torch.nn.Identity()
        elif last_layer_name_opt == 4:
            self.encoder.head.fc = torch.nn.Identity()
        elif last_layer_name_opt == 4:
            self.encoder.head.fc.fc2 = torch.nn.Identity()

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.linear(self.encoder(x))


def main(args: AttributeHashmap) -> None:
    '''
    Compute DSE and DSMI, and compute correlation with ImageNet acc.
    '''
    save_path_numpy = './results.npz'
    save_path_fig = './results'

    in_channels_map = {
        'mnist': 1,
        'cifar10': 3,
        'stl10': 3,
        'tinyimagenet': 3,
        'imagenet': 3,
    }
    num_classes_map = {
        'mnist': 10,
        'cifar10': 10,
        'stl10': 10,
        'tinyimagenet': 200,
        'imagenet': 1000,
    }
    dataset_dir_map = {
        'mnist': '/media/data1/chliu/mnist',
        'cifar10': '/media/data1/chliu/cifar10',
        'stl10': '/media/data1/chliu/stl10',
        'tinyimagenet': '/media/data1/chliu/tinyimagenet',
        'imagenet': '/media/data1/chliu/ImageNet',
    }
    args.in_channels = in_channels_map[args.dataset]
    args.num_classes = num_classes_map[args.dataset]
    args.dataset_dir = dataset_dir_map[args.dataset]

    # Load the tables for ImageNet accuracy (val and test set).
    df_val = pd.read_csv('./results-imagenet.csv')
    df_test = pd.read_csv('./results-imagenet-real.csv')

    df_val.drop([
        'top1_err', 'top5_err', 'crop_pct', 'interpolation', 'img_size',
        'param_count'
    ],
                axis=1,
                inplace=True)
    df_test.drop([
        'top1_err', 'top5_err', 'crop_pct', 'interpolation', 'top1_diff',
        'top5_diff', 'rank_diff'
    ],
                 axis=1,
                 inplace=True)
    df_val.rename(columns={
        'top1': 'val_acc_top1',
        'top5': 'val_acc_top5'
    },
                  inplace=True)
    df_test.rename(columns={
        'top1': 'test_acc_top1',
        'top5': 'test_acc_top5'
    },
                   inplace=True)
    df_combined = df_val.merge(df_test, on='model')
    del df_val, df_test

    # Iterate over all models and evaluate results.
    if os.path.isfile(save_path_numpy) and not args.restart:
        npz_file = np.load(save_path_numpy)
        results_dict = {
            'df_row_idx': int(npz_file['df_row_idx']),
            'model_params': list(npz_file['model_params']),
            'imagenet_val_acc_top1': list(npz_file['imagenet_val_acc_top1']),
            'imagenet_val_acc_top5': list(npz_file['imagenet_val_acc_top5']),
            'imagenet_test_acc_top1': list(npz_file['imagenet_test_acc_top1']),
            'imagenet_test_acc_top5': list(npz_file['imagenet_test_acc_top5']),
            'dse_Z': list(npz_file['dse_Z']),
            'cse_Z': list(npz_file['cse_Z']),
            'dsmi_Z_X': list(npz_file['dsmi_Z_X']),
            'csmi_Z_X': list(npz_file['csmi_Z_X']),
            'dsmi_Z_Y': list(npz_file['dsmi_Z_Y']),
            'csmi_Z_Y': list(npz_file['csmi_Z_Y']),
        }

    else:
        results_dict = {
            'df_row_idx': 0,
            'model_params': [],
            'imagenet_val_acc_top1': [],
            'imagenet_val_acc_top5': [],
            'imagenet_test_acc_top1': [],
            'imagenet_test_acc_top5': [],
            'dse_Z': [],
            'cse_Z': [],
            'dsmi_Z_X': [],
            'csmi_Z_X': [],
            'dsmi_Z_Y': [],
            'csmi_Z_Y': [],
        }

    # Iterate over the model candidates with pretrained weights.
    for df_row_idx, model_candidate in tqdm(df_combined.iterrows(),
                                            total=len(df_combined)):

        # This is for resuming progress.
        if df_row_idx < results_dict['df_row_idx']:
            continue
        results_dict['df_row_idx'] = df_row_idx

        args.imsize = model_candidate['img_size']
        results_dict['model_params'].append(
            float(model_candidate['param_count'].replace(',', '')))

        try:
            device = torch.device(
                'cuda:%d' %
                args.gpu_id if torch.cuda.is_available() else 'cpu')
            model = timm.create_model(model_name=model_candidate['model'],
                                      num_classes=args.num_classes,
                                      pretrained=False).to(device)
        except torch.cuda.OutOfMemoryError:
            device = torch.device('cpu')
            model = timm.create_model(model_name=model_candidate['model'],
                                      num_classes=args.num_classes,
                                      pretrained=False).to(device)

        try:
            model = ModelWithLatentAccess(model, num_classes=args.num_classes)
        except ThisArchitectureIsWeirdError as _:
            print('Cannot process: %s. Skipping it.' %
                  model_candidate['model'])
            continue

        model.eval()
        val_loader = get_val_loader(args=args)

        results_dict['imagenet_val_acc_top1'].append(
            model_candidate['val_acc_top1'])
        results_dict['imagenet_val_acc_top5'].append(
            model_candidate['val_acc_top5'])
        results_dict['imagenet_test_acc_top1'].append(
            model_candidate['test_acc_top1'])
        results_dict['imagenet_test_acc_top5'].append(
            model_candidate['test_acc_top5'])

        try:
            dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y = evaluate_dse_dsmi(
                args=args, val_loader=val_loader, model=model, device=device)
        except torch.cuda.OutOfMemoryError:
            device = torch.device('cpu')
            model = ModelWithLatentAccess(timm.create_model(
                model_name=model_candidate['model'],
                num_classes=args.num_classes,
                pretrained=False).to(device),
                                          num_classes=args.num_classes)
            dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y = evaluate_dse_dsmi(
                args=args, val_loader=val_loader, model=model, device=device)

        results_dict['dse_Z'].append(dse_Z)
        results_dict['cse_Z'].append(cse_Z)
        results_dict['dsmi_Z_X'].append(dsmi_Z_X)
        results_dict['csmi_Z_X'].append(csmi_Z_X)
        results_dict['dsmi_Z_Y'].append(dsmi_Z_Y)
        results_dict['csmi_Z_Y'].append(csmi_Z_Y)

        # It takes a long time to evaluate all models.
        # Plot and save results after each model evaluation.
        plot_figures(results_dict, save_path_fig=save_path_fig)
        with open(save_path_numpy, 'wb+') as f:
            np.savez(
                f,
                df_row_idx=df_row_idx,
                model_params=np.array(results_dict['model_params']),
                imagenet_val_acc_top1=np.array(
                    results_dict['imagenet_val_acc_top1']),
                imagenet_val_acc_top5=np.array(
                    results_dict['imagenet_val_acc_top5']),
                imagenet_test_acc_top1=np.array(
                    results_dict['imagenet_test_acc_top1']),
                imagenet_test_acc_top5=np.array(
                    results_dict['imagenet_test_acc_top5']),
                dse_Z=np.array(results_dict['dse_Z']),
                cse_Z=np.array(results_dict['cse_Z']),
                dsmi_Z_X=np.array(results_dict['dsmi_Z_X']),
                csmi_Z_X=np.array(results_dict['csmi_Z_X']),
                dsmi_Z_Y=np.array(results_dict['dsmi_Z_Y']),
                csmi_Z_Y=np.array(results_dict['csmi_Z_Y']),
            )

    return


@torch.no_grad()
def evaluate_dse_dsmi(args: AttributeHashmap,
                      val_loader: torch.utils.data.DataLoader,
                      model: torch.nn.Module, device: torch.device):

    tensor_X = None  # input
    tensor_Y = None  # label
    tensor_Z = None  # latent

    model.eval()
    for x, y_true in tqdm(val_loader):
        assert args.in_channels in [1, 3]
        if args.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x = x.repeat(1, 3, 1, 1)
        x, y_true = x.to(device), y_true.to(device)

        ## Record data for DSE and DSMI computation.

        # Downsample the input image to reduce memory usage.
        curr_X = torch.nn.functional.interpolate(
            x, size=(64, 64)).cpu().numpy().reshape(x.shape[0], -1)
        curr_Y = y_true.cpu().numpy()
        curr_Z = model.encode(x).cpu().numpy()
        if tensor_X is None:
            tensor_X, tensor_Y, tensor_Z = curr_X, curr_Y, curr_Z
        else:
            tensor_X = np.vstack((tensor_X, curr_X))
            tensor_Y = np.hstack((tensor_Y, curr_Y))
            tensor_Z = np.vstack((tensor_Z, curr_Z))

        if tensor_X.shape[0] > 5e3:
            # Only sample up to ~5k data points.
            break

    dse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z)
    cse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z,
                                       classic_shannon_entropy=True)
    dsmi_Z_X, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        n_clusters=args.num_classes)
    csmi_Z_X, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        n_clusters=args.num_classes,
        classic_shannon_entropy=True)

    dsmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z, reference_vectors=tensor_Y)
    csmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_Y,
        classic_shannon_entropy=True)

    return dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--restart',
        action='store_true',
        help=
        'If turned on, recompute from the first model. Otherwise resume where it left off.'
    )
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    seed_everything(args.random_seed)
    main(args)

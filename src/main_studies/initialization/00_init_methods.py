import argparse
import os
import sys
from typing import Tuple
from matplotlib import pyplot as plt

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm

from _timm_models import build_timm_model

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy
from dsmi import diffusion_spectral_mutual_information

sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from path_utils import update_config_dirs
from seed import seed_everything
from simclr import SingleInstanceTwoView
from extend import ExtendedDataset


def print_state_dict(state_dict: dict) -> str:
    state_str = ''
    for key in state_dict.keys():
        if '_loss' in key:
            try:
                state_str += '%s: %.6f. ' % (key, state_dict[key])
            except:
                state_str += '%s: %s. ' % (key, state_dict[key])
        else:
            try:
                state_str += '%s: %.3f. ' % (key, state_dict[key])
            except:
                state_str += '%s: %s. ' % (key, state_dict[key])
    return state_str


class CorruptLabelDataLoader(torch.utils.data.DataLoader):
    '''
    Randomly permute the labels such that there is an
    intentional mismatch between the images and labels.
    '''

    def __init__(self, dataloader, random_seed: int = None):
        self.dataloader = dataloader
        if 'targets' in self.dataloader.dataset.__dir__():
            # `targets` used in MNIST, CIFAR10, CIFAR100, ImageNet
            if random_seed is not None:
                np.random.seed(random_seed)
            self.dataloader.dataset.targets = np.random.permutation(
                self.dataloader.dataset.targets)
        elif 'labels' in self.dataloader.dataset.__dir__():
            # `labels` used in STL10
            if random_seed is not None:
                np.random.seed(random_seed)
            self.dataloader.dataset.labels = np.random.permutation(
                self.dataloader.dataset.labels)
        else:
            raise NotImplementedError(
                '`CorruptLabelDataLoader`: check the label name in dataset and update me.'
            )

    def __getattr__(self, name):
        return self.dataloader.__getattribute__(name)


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[
        torch.utils.data.DataLoader,
], AttributeHashmap]:
    if config.dataset == 'mnist':
        config.in_channels = 1
        config.num_classes = 10
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif config.dataset == 'cifar10':
        config.in_channels = 3
        config.num_classes = 10
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif config.dataset == 'stl10':
        config.in_channels = 3
        config.num_classes = 10
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    elif config.dataset == 'tinyimagenet':
        config.in_channels = 3
        config.num_classes = 200
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet

    elif config.dataset == 'imagenet':
        config.in_channels = 3
        config.num_classes = 1000
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    # NOTE: To accommodate the ViT models, we resize all images to 224x224.
    imsize = 224

    if config.method == 'supervised':
        if config.in_channels == 3:
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    imsize,
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.RandomResizedCrop(
                    imsize,
                    scale=(0.6, 1.6),
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(
                        brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
                ],
                                                   p=0.4),
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=dataset_mean,
                                                 std=dataset_std)
            ])
        else:
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    imsize,
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.RandomResizedCrop(
                    imsize,
                    scale=(0.6, 1.6),
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=dataset_mean,
                                                 std=dataset_std)
            ])

    elif config.method == 'simclr':
        transform_train = SingleInstanceTwoView(imsize=imsize,
                                                mean=dataset_mean,
                                                std=dataset_std)

    elif config.method == 'wronglabel':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(imsize,
                                          interpolation=torchvision.transforms.
                                          InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            imsize,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(imsize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    if config.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_val)

    elif config.dataset in ['stanfordcars', 'stl10', 'food101', 'flowers102']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            split='train',
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          split='test',
                                          download=True,
                                          transform=transform_val)

        if config.dataset == 'stl10' and config.method != 'wronglabel':
            # Training set has too few images (5000 images in total).
            # Let's augment it into a bigger dataset.
            train_dataset = ExtendedDataset(train_dataset,
                                            desired_len=10 *
                                            len(train_dataset))

    elif config.dataset in ['tinyimagenet', 'imagenet']:
        train_dataset = torchvision_dataset(config.dataset_dir,
                                            split='train',
                                            transform=transform_train)
        val_dataset = torchvision_dataset(config.dataset_dir,
                                          split='val',
                                          transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               num_workers=config.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             num_workers=config.num_workers,
                                             shuffle=False,
                                             pin_memory=True)
    if config.method == 'wronglabel':
        train_loader = CorruptLabelDataLoader(train_loader,
                                              random_seed=config.random_seed)

    if config.dataset == 'tinyimagenet' and config.method != 'wronglabel':
        # Validation set has too few images per class. Bad for DSE and DSMI estimation.
        # Therefore we extend it by a bit.
        val_dataset = torchvision_dataset(
            config.dataset_dir,
            split='val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    imsize,
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.RandomResizedCrop(
                    imsize,
                    scale=(0.6, 1.6),
                    interpolation=torchvision.transforms.InterpolationMode.
                    BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=dataset_mean,
                                                 std=dataset_std)
            ]))
        val_dataset = ExtendedDataset(val_dataset,
                                      desired_len=10 * len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=True,
            pin_memory=True)

    return (train_loader, val_loader), config


def run(config: AttributeHashmap) -> None:
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    train_loader, val_loader = dataloaders

    model = build_timm_model(model_name=config.model,
                             num_classes=config.num_classes).to(device)

    loss_fn_classification = torch.nn.CrossEntropyLoss()

    conv_init_std_list = []
    dse_Z_list = []
    # dsmi_Z_Y_list = []
    # dsmi_Z_X_list = []

    for conv_init_std in [1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 8e-2, 1e-1]:
        model.init_params(conv_init_std=conv_init_std)

        # Compute the results before training.
        val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, \
            dsmi_blockZ_Xs, dsmi_blockZ_Ys, precomputed_clusters_X = validate_epoch(
            config=config,
            val_loader=val_loader,
            model=model,
            device=device,
            loss_fn_classification=loss_fn_classification,
            precomputed_clusters_X=None)

        conv_init_std_list.append(conv_init_std)
        dse_Z_list.append(dse_Z)
        # dsmi_Z_Y_list.append(dsmi_Z_Y)
        # dsmi_Z_X_list.append(dsmi_Z_X)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['legend.fontsize'] = 12
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.plot(conv_init_std_list, dse_Z_list, marker='o', c='darkblue')
        ax.set_ylabel(r'DSE $S_D$(Z)', fontsize=12)
        ax.set_xlabel('Conv. Layer Initialization std.', fontsize=12)
        # ax = fig.add_subplot(2, 1, 2)
        # ax.spines[['right', 'top']].set_visible(False)
        # ax.plot(conv_init_std_list, dsmi_Z_X_list, marker='o', c='mediumblue')
        # ax.plot(conv_init_std_list, dsmi_Z_Y_list, marker='o', c='crimson')
        # ax.legend(['DSMI(Z; X)', 'DSMI(Z; Y)'])
        # ax.set_xlabel('Conv init std', fontsize=12)
        # ax.tick_params(axis='both', which='major', labelsize=12)
        fig.savefig('init_method_test')
    return


timm_model_blocks_map = {
    'resnet': ['layer1', 'layer2', 'layer3', 'layer4'],
    'resnext': ['layer1', 'layer2', 'layer3', 'layer4'],
    'convnext': ['stages', 4],
    'vit': ['blocks', 12],
    'swin': ['layers', 4],
    'xcit': ['blocks', 12],
}


def validate_epoch(config: AttributeHashmap,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module,
                   precomputed_clusters_X: np.array):

    correct, total_count_loss, total_count_acc = 0, 0, 0
    val_loss, val_acc = 0, 0

    tensor_X = None  # input
    tensor_Y = None  # label
    tensor_Z = None  # latent

    if config.block_by_block:
        '''Get block by block activations'''
        activation = {}

        def getActivation(name):

            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        handlers_list = []
        block_index_list = []
        # register forward hooks on key layers
        if config.model == 'resnet' or config.model == 'resnext':
            layers_names = timm_model_blocks_map[config.model]
            block_index_list = list(range(len(layers_names)))
            for i, layer_name in enumerate(layers_names):
                layer = getattr(model.encoder, layer_name)
                handlers_list.append(
                    layer.register_forward_hook(
                        getActivation('blocks_' + str(i))))
        else:
            main_blocks_name, block_cnt = timm_model_blocks_map[
                config.model][0], timm_model_blocks_map[config.model][1]
            block_index_list = list(range(block_cnt))

            for i in block_index_list:
                layer = getattr(model.encoder, main_blocks_name)[i]
                handlers_list.append(
                    layer.register_forward_hook(
                        getActivation('blocks_' + str(i))))

    model.eval()
    if config.block_by_block:
        blocks_features = [[] for _ in block_index_list]
    with torch.no_grad():
        for x, y_true in tqdm(val_loader):
            B = x.shape[0]
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = loss_fn_classification(y_pred, y_true)
            val_loss += loss.item() * B
            correct += torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()
            total_count_acc += B
            if config.method != 'simclr':
                total_count_loss += B

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

            if config.block_by_block:
                # Collect block activations from key layers
                for i in block_index_list:
                    curr_block_features = activation['blocks_' +
                                                     str(i)].cpu().numpy()
                    curr_block_features = curr_block_features.reshape(
                        curr_block_features.shape[0], -1)
                    blocks_features[i].append(curr_block_features)  # (B, D)

    if config.block_by_block:
        for i in block_index_list:
            blocks_features[i] = np.vstack(blocks_features[i])
            handlers_list[i].remove()

    if config.dataset == 'tinyimagenet':
        # For DSE, subsample for faster computation.
        dse_Z = diffusion_spectral_entropy(
            embedding_vectors=tensor_Z[:10000, :])
        cse_Z = diffusion_spectral_entropy(
            embedding_vectors=tensor_Z[:10000, :],
            classic_shannon_entropy=True)
    else:
        dse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z)
        cse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z,
                                           classic_shannon_entropy=True)

    dsmi_Z_X, precomputed_clusters_X = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        n_clusters=config.num_classes,
        precomputed_clusters=precomputed_clusters_X)
    csmi_Z_X, precomputed_clusters_X = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        n_clusters=config.num_classes,
        precomputed_clusters=precomputed_clusters_X,
        classic_shannon_entropy=True)

    dsmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z, reference_vectors=tensor_Y)
    csmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_Y,
        classic_shannon_entropy=True)

    dsmi_blockZ_Xs, dsmi_blockZ_Ys = [], []
    if config.block_by_block:
        for i in block_index_list:
            tensor_blockZ = blocks_features[i]
            dsmi_blockZ_X, _ = diffusion_spectral_mutual_information(
                embedding_vectors=tensor_blockZ,
                reference_vectors=tensor_X,
                precomputed_clusters=precomputed_clusters_X,
            )
            dsmi_blockZ_Xs.append(dsmi_blockZ_X)
            dsmi_blockZ_Y, _ = diffusion_spectral_mutual_information(
                embedding_vectors=tensor_blockZ, reference_vectors=tensor_Y)
            dsmi_blockZ_Ys.append(dsmi_blockZ_Y)

    if config.method == 'simclr':
        val_loss = torch.nan
    else:
        val_loss /= total_count_loss
    val_acc = correct / total_count_acc * 100

    return (val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y,
            csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, precomputed_clusters_X)


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
    parser.add_argument(
        '--block-by-block',
        action='store_true',
        help='If turned on, we compute the block-by-block DSE/DSMI.')
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    parser.add_argument(
        '--random-seed',
        help='Random Seed. If not None, will overwrite config.random_seed.',
        type=int,
        default=None)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.model = args.model
    config.block_by_block = args.block_by_block
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    config = update_config_dirs(AttributeHashmap(config))

    seed_everything(config.random_seed)

    run(config=config)

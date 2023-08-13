import argparse
import os
import sys
from typing import Tuple, Dict, Iterable
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-4])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy
from dsmi import diffusion_spectral_mutual_information

sys.path.insert(0, import_dir + '/src/nn/')
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from path_utils import update_config_dirs
from seed import seed_everything
from simclr import NTXentLoss, SingleInstanceTwoView
from scheduler import LinearWarmupCosineAnnealingLR
from timm_models import build_timm_model


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

    def __init__(self, dataloader):
        self.dataloader = dataloader
        if 'targets' in self.dataloader.dataset.__dir__():
            # `targets` used in MNIST, CIFAR10, CIFAR100
            np.random.seed(config.random_seed)
            self.dataloader.dataset.targets = np.random.permutation(
                self.dataloader.dataset.targets)
        elif 'labels' in self.dataloader.dataset.__dir__():
            # `labels` used in STL10
            np.random.seed(config.random_seed)
            self.dataloader.dataset.labels = np.random.permutation(
                self.dataloader.dataset.labels)

    def __getattr__(self, name):
        return self.dataloader.__getattribute__(name)


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
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
            torchvision.transforms.CenterCrop(imsize),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
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
        train_loader = CorruptLabelDataLoader(train_loader)

    return (train_loader, val_loader), config


def plot_figures(data_arrays: Dict[str, Iterable], save_path_fig: str) -> None:

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    # Plot of Entropy vs. epoch.
    fig = plt.figure(figsize=(40, 30))
    ax = fig.add_subplot(3, 2, 1)
    ax_secondary = ax.twinx()
    ax.spines[['right', 'top']].set_visible(False)
    ax_secondary.spines[['left', 'top']].set_visible(False)
    ln1 = ax.plot(data_arrays['epoch'],
                  data_arrays['cse_Z'],
                  c='grey',
                  linestyle='-.')
    ln2 = ax_secondary.plot(data_arrays['epoch'],
                            data_arrays['dse_Z'],
                            c='black')
    lns = ln1 + ln2
    ax.legend(lns, ['CSE(Z)', 'DSE(Z)'])
    ax.set_ylabel('Entropy', fontsize=40)
    ax.set_xlabel('Epochs Trained', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax_secondary.tick_params(axis='both', which='major', labelsize=30)

    # Plot of Entropy vs. Val. Acc.
    ax = fig.add_subplot(3, 2, 2)
    ax_secondary = ax.twinx()
    ax.spines[['right', 'top']].set_visible(False)
    ax_secondary.spines[['left', 'top']].set_visible(False)
    ln1 = ax.scatter(data_arrays['val_acc'],
                     data_arrays['cse_Z'],
                     c='grey',
                     alpha=0.5,
                     s=300)
    ln2 = ax_secondary.scatter(data_arrays['val_acc'],
                               data_arrays['dse_Z'],
                               c='black',
                               alpha=0.5,
                               s=300)
    lns = [ln1] + [ln2]
    ax.legend(lns, ['CSE(Z)', 'DSE(Z)'])
    ax.set_ylabel('Entropy', fontsize=40)
    ax.set_xlabel('Val. Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax_secondary.tick_params(axis='both', which='major', labelsize=30)

    # Display correlation.
    if len(data_arrays['val_acc']) > 1:
        ax.set_title(
            'CSE(Z), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['cse_Z'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['cse_Z'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['cse_Z'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['cse_Z'])[1]) +
            'DSE(Z), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['dse_Z'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['dse_Z'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['dse_Z'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['dse_Z'])[1]),
            fontsize=30)

    # Plot of Mutual Information vs. epoch.
    ax = fig.add_subplot(3, 2, 5)
    ax.spines[['right', 'top']].set_visible(False)
    # MI wrt Output
    ax.plot(data_arrays['epoch'],
            data_arrays['csmi_Z_Y'],
            c='grey',
            linestyle='-.')
    ax.plot(data_arrays['epoch'], data_arrays['dsmi_Z_Y'], c='black')
    # MI wrt Input
    ax.plot(data_arrays['epoch'],
            data_arrays['csmi_Z_X'],
            c='springgreen',
            linestyle='-.')
    ax.plot(data_arrays['epoch'], data_arrays['dsmi_Z_X'], c='darkgreen')
    ax.legend([
        'CSMI(Z; Y)',
        'DSMI(Z; Y)',
        'CSMI(Z; X)',
        'DSMI(Z; X)',
    ],
              loc='upper left')
    ax.set_ylabel('Mutual Information', fontsize=40)
    ax.set_xlabel('Epochs Trained', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Plot of Mutual Information vs. Val. Acc.
    ax = fig.add_subplot(3, 2, 6)
    ax.spines[['right', 'top']].set_visible(False)
    ax.scatter(data_arrays['val_acc'],
               data_arrays['csmi_Z_Y'],
               c='grey',
               alpha=0.5,
               s=300)
    ax.scatter(data_arrays['val_acc'],
               data_arrays['dsmi_Z_Y'],
               c='black',
               alpha=0.5,
               s=300)
    ax.scatter(data_arrays['val_acc'],
               data_arrays['csmi_Z_X'],
               c='springgreen',
               alpha=0.5,
               s=300)
    ax.scatter(data_arrays['val_acc'],
               data_arrays['dsmi_Z_X'],
               c='darkgreen',
               alpha=0.5,
               s=300)
    ax.legend([
        'CSMI(Z; Y)',
        'DSMI(Z; Y)',
        'CSMI(Z; X)',
        'DSMI(Z; X)',
    ],
              loc='upper left')
    ax.set_ylabel('Mutual Information', fontsize=40)
    ax.set_xlabel('Downstream Classification Accuracy', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # # Display correlation.
    if len(data_arrays['val_acc']) > 1:
        ax.set_title(
            'CSMI(Z; Y), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['csmi_Z_Y'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['csmi_Z_Y'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['csmi_Z_Y'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['csmi_Z_Y'])[1]) +
            'DSMI(Z; Y), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['dsmi_Z_Y'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['dsmi_Z_Y'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['dsmi_Z_Y'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['dsmi_Z_Y'])[1]) +
            'CSMI(Z; X), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['csmi_Z_X'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['csmi_Z_X'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['csmi_Z_X'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['csmi_Z_X'])[1]) +
            'DSMI(Z; X), P.R: %.3f (p = %.4f), S.R: %.3f (p = %.4f);\n' %
            (pearsonr(data_arrays['val_acc'], data_arrays['dsmi_Z_X'])[0],
             pearsonr(data_arrays['val_acc'], data_arrays['dsmi_Z_X'])[1],
             spearmanr(data_arrays['val_acc'], data_arrays['dsmi_Z_X'])[0],
             spearmanr(data_arrays['val_acc'], data_arrays['dsmi_Z_X'])[1]),
            fontsize=30)
    fig.tight_layout()
    fig.savefig(save_path_fig)
    plt.close(fig=fig)

    # Plot I(Z; Y) vs. I(Z; X) for each epoch and each block.
    fig2 = plt.figure(figsize=(20, 20))
    ax = fig2.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    dsmi_blockZ_X_list = data_arrays['dsmi_blockZ_X_list']
    dsmi_blockZ_Y_list = data_arrays['dsmi_blockZ_Y_list']
    colors = plt.cm.jet(np.linespace(0, 1, len(dsmi_blockZ_X_list))
                        
    for i in range(dsmi_blockZ_X_list):
        ax.scatter(dsmi_blockZ_X_list[i], dsmi_blockZ_Y_list[i], c=colors[i])
        ax.set_xlabel('I(Z; X)', fontsize=40)
        ax.set_ylabel('I(Z; Y)', fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=30)
        ax.plot(dsmi_blockZ_X_list[i], dsmi_blockZ_Y_list[i], c=colors[i], alpha=0.3)

        # show color map
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Epoch', fontsize=30)
        cbar.ax.tick_params(labelsize=30)

    save_path_fig2 = '%s-blocks' % (save_path_fig)
    fig2.tight_layout()
    fig2.savefig(save_path_fig2)
    plt.close(fig=fig2)

    return


def train(config: AttributeHashmap) -> None:
    '''
    Train our simple model and record the checkpoints along the training process.
    '''
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    train_loader, val_loader = dataloaders

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = '%s/%s-%s-%s-seed%s.log' % (config.log_dir, config.dataset,
                                           config.method, config.model,
                                           config.random_seed)
    save_path_fig = '%s/%s-%s-%s-seed%s' % (config.log_dir, config.dataset,
                                            config.method, config.model,
                                            config.random_seed)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=log_path, to_console=False)

    model = build_timm_model(model_name=config.model,
                             num_classes=config.num_classes).to(device)
    model.init_params()

    loss_fn_classification = torch.nn.CrossEntropyLoss()
    loss_fn_simclr = NTXentLoss()

    # `val_metric` is val acc for good training,
    # whereas train/val acc divergence for wrong label training.
    if config.method == 'wronglabel':
        val_metric = 'acc_diverg'
    else:
        val_metric = 'val_acc'

    # Compute the results before training.
    val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, precomputed_clusters_X = validate_epoch(
        config=config,
        val_loader=val_loader,
        model=model,
        device=device,
        loss_fn_classification=loss_fn_classification,
        precomputed_clusters_X=None)

    state_dict = {
        'train_loss': 'Not started',
        'train_acc': 'Not started',
        'val_loss': val_loss,
        'val_acc': val_acc,
        'acc_diverg': 'Not started',
    }
    log('Epoch: %d. %s' % (0, print_state_dict(state_dict)),
        filepath=log_path,
        to_console=False)

    results_dict = {
        'epoch': [0],
        'dse_Z': [dse_Z],
        'cse_Z': [cse_Z],
        'dsmi_Z_X': [dsmi_Z_X],
        'csmi_Z_X': [csmi_Z_X],
        'dsmi_Z_Y': [dsmi_Z_Y],
        'csmi_Z_Y': [csmi_Z_Y],
        'val_acc': [0],
        'dsmi_blockZ_Xs': [np.array(dsmi_blockZ_Xs)],
        'dsmi_blockZ_Ys': [np.array(dsmi_blockZ_Ys)],
    }

    if config.method in ['supervised', 'wronglabel']:
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.linear.parameters()),
                                lr=float(config.learning_rate))
    elif config.method == 'simclr':
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.projection_head.parameters()),
                                lr=float(config.learning_rate))

    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=opt,
                                                 warmup_epochs=min(
                                                     10,
                                                     config.max_epoch // 5),
                                                 max_epochs=config.max_epoch)

    best_val_metric = 0
    best_model = None

    val_metric_pct_list = [20, 30, 40, 50, 60, 70, 80, 90]
    is_model_saved = {}
    for val_metric_pct in val_metric_pct_list:
        is_model_saved[str(val_metric_pct)] = False

    for epoch_idx in tqdm(range(1, config.max_epoch)):
        # For SimCLR, only perform validation / linear probing every 5 epochs.
        skip_epoch_simlr = epoch_idx % 5 != 0

        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
            'acc_diverg': 0,
        }

        if config.method == 'simclr':
            state_dict['train_simclr_pseudoAcc'] = 0

        #
        '''
        Training
        '''
        model.train()
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler.step()
        correct, total_count_loss, total_count_acc = 0, 0, 0
        for _, (x, y_true) in enumerate(tqdm(train_loader)):
            if config.method in ['supervised', 'wronglabel']:
                # Not using contrastive learning.

                B = x.shape[0]
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x = x.repeat(1, 3, 1, 1)
                x, y_true = x.to(device), y_true.to(device)

                y_pred = model(x)
                loss = loss_fn_classification(y_pred, y_true)
                state_dict['train_loss'] += loss.item() * B
                correct += torch.sum(
                    torch.argmax(y_pred, dim=-1) == y_true).item()
                total_count_loss += B
                total_count_acc += B

                opt.zero_grad()
                loss.backward()
                opt.step()

            elif config.method == 'simclr':
                # Using SimCLR.

                x_aug1, x_aug2 = x
                B = x_aug1.shape[0]
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x_aug1 = x_aug1.repeat(1, 3, 1, 1)
                    x_aug2 = x_aug2.repeat(1, 3, 1, 1)
                x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
                    device), y_true.to(device)

                # Train encoder.
                z1 = model.project(x_aug1)
                z2 = model.project(x_aug2)

                loss, pseudo_acc = loss_fn_simclr(z1, z2)
                state_dict['train_loss'] += loss.item() * B
                state_dict['train_simclr_pseudoAcc'] += pseudo_acc * B
                total_count_loss += B

                opt.zero_grad()
                loss.backward()
                opt.step()

        if config.method == 'simclr':
            state_dict['train_simclr_pseudoAcc'] /= total_count_loss
        else:
            state_dict['train_acc'] = correct / total_count_acc * 100
        state_dict['train_loss'] /= total_count_loss

        #
        '''
        Validation (or Linear Probing + Validation)
        '''
        if config.method == 'simclr':
            if not skip_epoch_simlr:
                # This function call includes validation.
                probing_acc, val_acc_final, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, _ = linear_probing(
                    config=config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model=model,
                    device=device,
                    loss_fn_classification=loss_fn_classification,
                    precomputed_clusters_X=precomputed_clusters_X)
                state_dict['train_acc'] = probing_acc
                state_dict['val_loss'] = np.nan
                state_dict['val_acc'] = val_acc_final
            else:
                state_dict['train_acc'] = 'Val skipped for efficiency'
                state_dict['val_loss'] = 'Val skipped for efficiency'
                state_dict['val_acc'] = 'Val skipped for efficiency'
        else:
            val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, _ = validate_epoch(
                config=config,
                val_loader=val_loader,
                model=model,
                device=device,
                loss_fn_classification=loss_fn_classification,
                precomputed_clusters_X=precomputed_clusters_X)
            state_dict['val_loss'] = val_loss
            state_dict['val_acc'] = val_acc

        if not (config.method == 'simclr' and skip_epoch_simlr):
            state_dict['acc_diverg'] = \
                state_dict['train_acc'] - state_dict['val_acc']
        else:
            state_dict['acc_diverg'] = 'Val skipped for efficiency'

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        if not (config.method == 'simclr' and skip_epoch_simlr):
            results_dict['epoch'].append(epoch_idx)
            results_dict['dse_Z'].append(dse_Z)
            results_dict['cse_Z'].append(cse_Z)
            results_dict['dsmi_Z_X'].append(dsmi_Z_X)
            results_dict['csmi_Z_X'].append(csmi_Z_X)
            results_dict['dsmi_Z_Y'].append(dsmi_Z_Y)
            results_dict['csmi_Z_Y'].append(csmi_Z_Y)
            results_dict['val_acc'].append(state_dict['val_acc'])
            results_dict['dsmi_blockZ_X_list'].append(np.array(dsmi_blockZ_Xs))
            results_dict['dsmi_blockZ_Y_list'].append(np.array(dsmi_blockZ_Ys))

        plot_figures(data_arrays=results_dict, save_path_fig=save_path_fig)

        # Save best model
        if not (config.method == 'simclr' and skip_epoch_simlr):
            if state_dict[val_metric] > best_val_metric:
                best_val_metric = state_dict[val_metric]
                best_model = model.state_dict()
                model_save_path = '%s/%s-%s-%s-seed%s-%s' % (
                    config.checkpoint_dir, config.dataset, config.method,
                    config.model, config.random_seed, '%s_best.pth' % val_metric)
                torch.save(best_model, model_save_path)
                log('Best model (so far) successfully saved.',
                    filepath=log_path,
                    to_console=False)

                # Save model at each percentile.
                for val_metric_pct in val_metric_pct_list:
                    if state_dict[val_metric] > val_metric_pct and \
                    not is_model_saved[str(val_metric_pct)]:
                        model_save_path = '%s/%s-%s-%s-seed%s-%s' % (
                            config.checkpoint_dir, config.dataset, config.method,
                            config.model, config.random_seed, '%s_%s%%.pth' %
                            (val_metric, val_metric_pct))
                        torch.save(best_model, model_save_path)
                        is_model_saved[str(val_metric_pct)] = True
                        log('%s:%s%% model successfully saved.' %
                            (val_metric, val_metric_pct),
                            filepath=log_path,
                            to_console=False)

    # Save the results after training.
    save_path_numpy = '%s/%s-%s-%s-seed%s/%s' % (
        config.output_save_path, config.dataset, config.method, config.model,
        config.random_seed, 'results.npz')
    os.makedirs(os.path.dirname(save_path_numpy), exist_ok=True)

    with open(save_path_numpy, 'wb+') as f:
        np.savez(
            f,
            epoch=np.array(results_dict['epoch']),
            val_acc=np.array(results_dict['val_acc']),
            dse_Z=np.array(results_dict['dse_Z']),
            cse_Z=np.array(results_dict['cse_Z']),
            dsmi_Z_X=np.array(results_dict['dsmi_Z_X']),
            csmi_Z_X=np.array(results_dict['csmi_Z_X']),
            dsmi_Z_Y=np.array(results_dict['dsmi_Z_Y']),
            csmi_Z_Y=np.array(results_dict['csmi_Z_Y']),
        )
    
    # Save block by block DSMI results
    save_path_numpy = '%s/%s-%s-%s-seed%s/%s' % (
        config.output_save_path, config.dataset, config.method, config.model,
        config.random_seed, 'block-results.npz')
    os.makedirs(os.path.dirname(save_path_numpy), exist_ok=True)

    with open(save_path_numpy, 'wb+') as f:
        np.savez(
            f,
            epoch=np.array(results_dict['epoch']),
            dsmi_blockZ_X_list=results_dict['dsmi_blockZ_X_list'],
            dsmi_blockZ_Y_list=results_dict['dsmi_blockZ_Y_list'],
        )

    return

timm_model_blocks_map = {
    'resnet': ['layer1', 'layer2', 'layer3', 'layer4'],
    'resnext': ['layer1', 'layer2', 'layer3', 'layer4'],
    'mobilenet': ['blocks', 5],
    'vit': ['blocks', 12],
    'swin': ['layers', 4],
    'mobilevit': ['stages', 5],
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
            handlers_list.append(layer.register_forward_hook(getActivation('blocks_'+str(i))))
    else:
        main_blocks_name, block_cnt = timm_model_blocks_map[config.model][0], timm_model_blocks_map[config.model][1]
        block_index_list = list(range(block_cnt))
        
        for i in block_index_list:
            layer = getattr(model.encoder, main_blocks_name)[i]
            handlers_list.append(layer.register_forward_hook(getActivation('blocks_'+str(i))))


    model.eval()
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
            
            # Collect block activations from key layers
            for i in block_index_list:
                curr_block_features = activation['blocks_'+str(i)].cpu().numpy()
                curr_block_features = curr_block_features.reshape(curr_block_features.shape[0], -1)
                blocks_features[i].append(curr_block_features) # (B, D)
    for i in block_index_list:
        blocks_features[i] = np.vstack(blocks_features[i])
        handlers_list[i].remove()

    dse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z)
    cse_Z = diffusion_spectral_entropy(embedding_vectors=tensor_Z,
                                       classic_shannon_entropy=True)
    dsmi_Z_X, precomputed_clusters_X = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        precomputed_clusters=precomputed_clusters_X)
    csmi_Z_X, precomputed_clusters_X = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_X,
        precomputed_clusters=precomputed_clusters_X,
        classic_shannon_entropy=True)

    dsmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z, reference_vectors=tensor_Y)
    csmi_Z_Y, _ = diffusion_spectral_mutual_information(
        embedding_vectors=tensor_Z,
        reference_vectors=tensor_Y,
        classic_shannon_entropy=True)
    
    dsmi_blockZ_X_list, dsmi_blockZ_Y_list = [], []
    for i in block_index_list:
        tensor_blockZ = blocks_features[i]
        dsmi_blockZ_X, _ = diffusion_spectral_mutual_information(
            embedding_vectors=tensor_blockZ,
            reference_vectors=tensor_X,
            precomputed_clusters=precomputed_clusters_X,
        )
        dsmi_blockZ_X_list.append(dsmi_blockZ_X)
        dsmi_blockZ_Y, _ = diffusion_spectral_mutual_information(
            embedding_vectors=tensor_blockZ,
            reference_vectors=tensor_Y
        )
        dsmi_blockZ_Y_list.append(dsmi_blockZ_Y)

    if config.method == 'simclr':
        val_loss = torch.nan
    else:
        val_loss /= total_count_loss
    val_acc = correct / total_count_acc * 100

    return (val_loss, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, 
            dsmi_blockZ_X_list, dsmi_blockZ_Y_list, precomputed_clusters_X)


def linear_probing(config: AttributeHashmap,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module,
                   precomputed_clusters_X: np.array):

    # Separately train linear classifier.
    model.init_linear()
    # Note: Need to create another optimizer because the model will keep updating
    # even after freezing with `requires_grad = False` when `opt` has `momentum`.
    opt_probing = torch.optim.AdamW(list(model.linear.parameters()),
                                    lr=float(config.learning_rate_probing))

    lr_scheduler_probing = LinearWarmupCosineAnnealingLR(
        optimizer=opt_probing,
        warmup_epochs=min(10, config.probing_epoch // 5),
        max_epochs=config.probing_epoch)

    for _ in tqdm(range(config.probing_epoch)):
        # Because of linear warmup, first step has zero LR. Hence step once before training.
        lr_scheduler_probing.step()
        probing_acc = linear_probing_epoch(
            config=config,
            train_loader=train_loader,
            model=model,
            device=device,
            opt_probing=opt_probing,
            loss_fn_classification=loss_fn_classification)

    _, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, dsmi_blockZ_Xs, dsmi_blockZ_Ys, _ = validate_epoch(
        config=config,
        val_loader=val_loader,
        model=model,
        device=device,
        loss_fn_classification=loss_fn_classification,
        precomputed_clusters_X=precomputed_clusters_X)

    return probing_acc, val_acc, dse_Z, cse_Z, dsmi_Z_X, csmi_Z_X, dsmi_Z_Y, csmi_Z_Y, precomputed_clusters_X


def linear_probing_epoch(config: AttributeHashmap,
                         train_loader: torch.utils.data.DataLoader,
                         model: torch.nn.Module, device: torch.device,
                         opt_probing: torch.optim.Optimizer,
                         loss_fn_classification: torch.nn.Module):
    model.train()
    correct, total_count_acc = 0, 0
    for _, (x, y_true) in enumerate(train_loader):
        x_aug1, x_aug2 = x
        B = x_aug1.shape[0]
        assert config.in_channels in [1, 3]
        if config.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x_aug1 = x_aug1.repeat(1, 3, 1, 1)
            x_aug2 = x_aug2.repeat(1, 3, 1, 1)
        x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug2.to(
            device), y_true.to(device)

        with torch.no_grad():
            h1, h2 = model.encode(x_aug1), model.encode(x_aug2)
        y_pred_aug1, y_pred_aug2 = model.linear(h1), model.linear(h2)
        loss_aug1 = loss_fn_classification(y_pred_aug1, y_true)
        loss_aug2 = loss_fn_classification(y_pred_aug2, y_true)
        loss = (loss_aug1 + loss_aug2) / 2
        correct += torch.sum(
            torch.argmax(y_pred_aug1, dim=-1) == y_true).item()
        correct += torch.sum(
            torch.argmax(y_pred_aug2, dim=-1) == y_true).item()
        total_count_acc += 2 * B

        opt_probing.zero_grad()
        loss.backward()
        opt_probing.step()

    probing_acc = correct / total_count_acc * 100

    return probing_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument(
        '--model',
        help='model name: [resnet, resnext, mobilenet, vit, swin, mobilevit]',
        type=str,
        required=True)
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
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    config = update_config_dirs(AttributeHashmap(config))

    # Update checkpoint dir.
    config.checkpoint_dir = '%s/%s-%s-%s-seed%s/' % (
        config.checkpoint_dir, config.dataset, config.method, config.model,
        config.random_seed)

    seed_everything(config.random_seed)

    train(config=config)

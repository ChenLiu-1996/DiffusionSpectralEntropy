import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm

from api.dse import diffusion_spectral_entropy
from api.dsmi import diffusion_spectral_mutual_information

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/nn/')
sys.path.insert(0, import_dir + '/utils/')
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
            state_str += '%s: %.6f. ' % (key, state_dict[key])
        else:
            state_str += '%s: %.3f. ' % (key, state_dict[key])
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
            torchvision.transforms.CenterCrop(imsize),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])

    transform_val = torchvision.transforms.Compose([
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

    # Compute the results before training.
    val_loss, val_acc = validate_epoch(
        config=config,
        val_loader=val_loader,
        model=model,
        device=device,
        loss_fn_classification=loss_fn_classification)

    if config.method in ['supervised', 'wronglabel']:
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.linear.parameters()),
                                lr=float(config.learning_rate),
                                weight_decay=float(config.weight_decay))
    elif config.method == 'simclr':
        opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                                list(model.projection_head.parameters()),
                                lr=float(config.learning_rate),
                                weight_decay=float(config.weight_decay))

    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=opt,
                                                 warmup_epochs=min(
                                                     10,
                                                     config.max_epoch // 5),
                                                 max_epochs=config.max_epoch)

    # `val_metric` is val acc for good training,
    # whereas train/val acc divergence for wrong label training.
    if config.method == 'wronglabel':
        val_metric = 'acc_diverg'
    else:
        val_metric = 'val_acc'
    best_val_metric = 0
    best_model = None

    val_metric_pct_list = [20, 30, 40, 50, 60, 70, 80, 90]
    is_model_saved = {}
    for val_metric_pct in val_metric_pct_list:
        is_model_saved['%s_%s%%' % (val_metric, val_metric_pct)] = False

    for epoch_idx in tqdm(range(config.max_epoch)):
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
        correct, total_count_loss, total_count_acc = 0, 0, 0
        for _, (x, y_true) in enumerate(train_loader):
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

        lr_scheduler.step()

        #
        '''
        Validation (or Linear Probing + Validation)
        '''
        if config.method == 'simclr':
            # This function call includes validation.
            probing_acc, val_acc_final = linear_probing(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                device=device,
                loss_fn_classification=loss_fn_classification)
            state_dict['train_acc'] = probing_acc
            state_dict['val_loss'] = np.nan
            state_dict['val_acc'] = val_acc_final
        else:
            val_loss, val_acc = validate_epoch(
                config=config,
                val_loader=val_loader,
                model=model,
                device=device,
                loss_fn_classification=loss_fn_classification)
            state_dict['val_loss'] = val_loss
            state_dict['val_acc'] = val_acc

        state_dict['acc_diverg'] = \
            state_dict['train_acc'] - state_dict['val_acc']

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        # Save best model
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
                   not is_model_saved['%s_%s%%' % (val_metric, val_metric_pct)]:
                    model_save_path = '%s/%s-%s-%s-seed%s-%s' % (
                        config.checkpoint_dir, config.dataset, config.method,
                        config.model, config.random_seed, '%s_%s%%.pth' %
                        (val_metric, val_metric_pct))
                    torch.save(best_model, model_save_path)
                    is_model_saved['val_acc_%s%%' % val_metric] = True
                    log('%s%% accuracy model successfully saved.' % val_metric,
                        filepath=log_path,
                        to_console=False)

    return


def validate_epoch(config: AttributeHashmap,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module):

    correct, total_count_loss, total_count_acc = 0, 0, 0
    val_loss, val_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for x, y_true in val_loader:
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

    if config.method == 'simclr':
        val_loss = torch.nan
    else:
        val_loss /= total_count_loss
    val_acc = correct / total_count_acc * 100

    return val_loss, val_acc


def linear_probing(config: AttributeHashmap,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module):

    # Separately train linear classifier.
    model.init_linear()
    # Note: Need to create another optimizer because the model will keep updating
    # even after freezing with `requires_grad = False` when `opt` has `momentum`.
    opt_probing = torch.optim.AdamW(list(model.linear.parameters()),
                                    lr=float(config.learning_rate_probing),
                                    weight_decay=float(config.weight_decay))

    lr_scheduler_probing = LinearWarmupCosineAnnealingLR(
        optimizer=opt_probing,
        warmup_epochs=min(10, config.probing_epoch // 5),
        max_epochs=config.probing_epoch)

    for _ in range(config.probing_epoch):
        probing_acc = linear_probing_epoch(
            config=config,
            train_loader=train_loader,
            model=model,
            device=device,
            opt_probing=opt_probing,
            loss_fn_classification=loss_fn_classification)
        lr_scheduler_probing.step()

    _, val_acc = validate_epoch(config=config,
                                val_loader=val_loader,
                                model=model,
                                device=device,
                                loss_fn_classification=loss_fn_classification)

    return probing_acc, val_acc


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
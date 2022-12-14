import argparse
import os
import sys
from typing import Tuple

import torch
import torchvision
import yaml
from models import FlexibleResNet50
from simclr import SingleInstanceTwoView
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from seed import seed_everything


def update_config_dirs(config: AttributeHashmap) -> AttributeHashmap:
    root_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in config.keys():
        if type(config[key]) is str and '$ROOT_DIR' in config[key]:
            config[key] = config[key].replace('$ROOT_DIR', root_dir)
    return config


def print_state_dict(state_dict: dict) -> str:
    state_str = ''
    for key in state_dict.keys():
        if '_loss' in key:
            state_str += '%s: %.6f. ' % (key, state_dict[key])
        else:
            state_str += '%s: %.3f. ' % (key, state_dict[key])
    return state_str


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
    if config.dataset == 'mnist':
        config.in_channels = 1
        config.num_classes = 10
        imsize = 32
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset_loader = torchvision.datasets.MNIST

    elif config.dataset == 'cifar10':
        config.in_channels = 3
        config.num_classes = 10
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset_loader = torchvision.datasets.CIFAR10

    elif config.dataset == 'cifar100':
        config.in_channels = 3
        config.num_classes = 100
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset_loader = torchvision.datasets.CIFAR100

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    if config.contrastive == 'NA':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(imsize, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])
        transform_val = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])

    elif config.contrastive == 'simclr':
        transform_train = SingleInstanceTwoView(imsize=imsize,
                                                mean=dataset_mean,
                                                std=dataset_std)
        transform_val = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])

    train_loader = torch.utils.data.DataLoader(torchvision_dataset_loader(
        config.dataset_dir,
        train=True,
        download=True,
        transform=transform_train),
                                               batch_size=config.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(torchvision_dataset_loader(
        config.dataset_dir,
        train=False,
        download=True,
        transform=transform_val),
                                             batch_size=config.batch_size,
                                             shuffle=False)

    return (train_loader, val_loader), config


def train(config: AttributeHashmap) -> None:
    '''
    Trains our simple model and record the checkpoints along the training process.
    '''
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = '%s/%s-%s.log' % (config.log_dir, config.dataset,
                                 config.contrastive)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=log_path, to_console=False)

    model = FlexibleResNet50(contrastive=config.contrastive != 'NA',
                             num_classes=config.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(config.learning_rate),
                            weight_decay=float(config.weight_decay))

    train_loader, val_loader = dataloaders

    loss_fn_classification = torch.nn.CrossEntropyLoss()

    is_model_saved = {
        'val_acc_50%': False,
        'val_acc_70%': False,
    }
    best_val_acc = 0
    best_model = None

    for epoch_idx in tqdm(range(config.max_epoch)):
        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
        }

        model.train()
        correct, total = 0, 0
        for x, y_true in train_loader:

            if config.contrastive == 'NA':
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
                total += B

                opt.zero_grad()
                loss.backward()
                opt.step()

            elif config.contrastive == 'simclr':
                import pdb
                pdb.set_trace()
                # Using SimCLR.
                # x_aug1 = augment(x, random_seed=config.random_seed)
                # x_aug2 = augment(x, random_seed=config.random_seed)
                raise NotImplementedError

        state_dict['train_loss'] /= total
        state_dict['train_acc'] = correct / total * 100

        model.eval()
        for x, y_true in val_loader:
            if config.contrastive == 'NA':
                # Not using contrastive learning.
                B = x.shape[0]
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x = x.repeat(1, 3, 1, 1)
                x, y_true = x.to(device), y_true.to(device)

                y_pred = model(x)
                loss = loss_fn_classification(y_pred, y_true)
                state_dict['val_loss'] += loss.item() * B
                correct += torch.sum(
                    torch.argmax(y_pred, dim=-1) == y_true).item()
                total += B

            elif config.contrastive == 'simclr':
                # Using SimCLR.
                raise NotImplementedError

        state_dict['val_loss'] /= total
        state_dict['val_acc'] = correct / total * 100

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        if state_dict['val_acc'] > best_val_acc:
            best_model = model.state_dict()
            model_save_path = '%s/%s-%s-%s' % (
                config.checkpoint_dir, config.dataset, config.contrastive,
                'best_val_acc.pth')
            torch.save(best_model, model_save_path)

            if state_dict['val_acc'] > 50 and not is_model_saved['val_acc_50%']:
                model_save_path = '%s/%s-%s-%s' % (
                    config.checkpoint_dir, config.dataset, config.contrastive,
                    'val_acc_50%.pth')
                torch.save(best_model, model_save_path)
                is_model_saved['val_acc_50%'] = True

            if state_dict['val_acc'] > 70 and not is_model_saved['val_acc_70%']:
                model_save_path = '%s/%s-%s-%s' % (
                    config.checkpoint_dir, config.dataset, config.contrastive,
                    'val_acc_70%.pth')
                torch.save(best_model, model_save_path)
                is_model_saved['val_acc_70%'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = update_config_dirs(AttributeHashmap(config))

    seed_everything(config.random_seed)
    train(config=config)

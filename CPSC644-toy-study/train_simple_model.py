import argparse
import os
import sys
from typing import Tuple

import torch
import torchvision
import yaml
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from seed import seed_everything


def update_config_dirs(config: AttributeHashmap) -> AttributeHashmap:
    root_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    for key in config.keys():
        if type(config[key]) is str and '$ROOT_DIR' in config[key]:
            config[key] = config[key].replace('$ROOT_DIR', root_dir)
    return config


def get_dataloaders(
    config: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:
    if config.dataset == 'mnist':
        config.in_channels = 1
        config.num_classes = 10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(config.dataset_dir,
                                       train=True,
                                       download=True,
                                       transform=transform),
            batch_size=config.batch_size,
            shuffle=True)
        validation_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(config.dataset_dir,
                                       train=False,
                                       download=True,
                                       transform=transform),
            batch_size=config.batch_size,
            shuffle=False)

    elif config.dataset == 'cifar10':
        config.in_channels = 3
        config.num_classes = 10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(config.dataset_dir,
                                         train=True,
                                         download=True,
                                         transform=transform),
            batch_size=config.batch_size,
            shuffle=True)
        validation_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(config.dataset_dir,
                                         train=False,
                                         download=True,
                                         transform=transform),
            batch_size=config.batch_size,
            shuffle=False)

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    return (train_loader, validation_loader), config


def train(config: AttributeHashmap) -> None:
    '''
    Trains our simple model and record the checkpoints along the training process.
    '''
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)

    model = torchvision.models.resnet50(
        num_classes=config.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(config.learning_rate),
                            weight_decay=float(config.weight_decay))

    train_loader, val_loader = dataloaders

    loss_fn_classification = torch.nn.CrossEntropyLoss()

    for _ in tqdm(range(config.max_epoch)):
        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
        }

        for x, y_true in train_loader:
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)

            x = x.to(device)
            y_true = y_true.to(device)

            model.train()
            if config.contrastive == 'NA':
                # Not using contrastive learning.
                y_pred = model(x)
                loss = loss_fn_classification(y_pred, y_true)
                state_dict['train_loss'] += loss.item() / len(train_loader)
                import pdb
                pdb.set_trace()
                state_dict['train_acc'] += torch.argmax(
                    y_pred) == y_true / len(train_loader)

                opt.zero_grad()
                loss.backward()
                opt.step()

            elif config.contrastive == 'simclr':
                # Using SimCLR.
                raise NotImplementedError

        for x, y_true in val_loader:
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)

            x = x.to(device)
            y_true = y_true.to(device)

            model.eval()
            if config.contrastive == 'NA':
                # Not using contrastive learning.
                y_pred = model(x)
                loss = loss_fn_classification(y_pred, y_true)
                state_dict['val_loss'] += loss.item() / len(val_loader)

            elif config.contrastive == 'simclr':
                # Using SimCLR.
                raise NotImplementedError

        print(state_dict)


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

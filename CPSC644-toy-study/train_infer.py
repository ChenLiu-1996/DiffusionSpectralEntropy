import argparse
import os
import sys
from glob import glob
from typing import Tuple

import numpy as np
import torch
import torchvision
import yaml
from models import ResNet50
from simclr import NTXentLoss, SingleInstanceTwoView
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap
from early_stop import EarlyStopping
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
    Train our simple model and record the checkpoints along the training process.
    '''
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    train_loader, val_loader = dataloaders

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

    model = ResNet50(num_classes=config.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(),
                            lr=float(config.learning_rate),
                            weight_decay=float(config.weight_decay))

    loss_fn_classification = torch.nn.CrossEntropyLoss()
    loss_fn_simclr = NTXentLoss()
    early_stopper = EarlyStopping(mode='max',
                                  patience=config.patience,
                                  percentage=False)

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

        correct, total = 0, 0
        simclr_stage1_initialized, simclr_stage2_initialized = False, False
        model.train()
        for batch_idx, (x, y_true) in enumerate(train_loader):

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
                # Using SimCLR.

                x_aug1, x_aug2 = x
                B = x_aug1.shape[0]
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x_aug1 = x_aug1.repeat(1, 3, 1, 1)
                    x_aug2 = x_aug2.repeat(1, 3, 1, 1)
                x_aug1, x_aug2, y_true = x_aug1.to(device), x_aug1.to(
                    device), y_true.to(device)

                if batch_idx < 0.8 * len(train_loader):
                    # Freeze linear classifier, train encoder.
                    if not simclr_stage1_initialized:
                        model.freeze_linear()
                        model.unfreeze_encoder()
                        simclr_stage1_initialized = True

                    z1 = model.encode(x_aug1)
                    z2 = model.encode(x_aug2)

                    loss = loss_fn_simclr(z1, z2)
                    state_dict['train_loss'] += loss.item() * B

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                else:
                    # Freeze encoder, train linear classifier.
                    if not simclr_stage2_initialized:
                        model.init_linear()
                        model.unfreeze_linear()
                        model.freeze_encoder()
                        simclr_stage2_initialized = True

                    y_pred = model(x_aug1)
                    correct += torch.sum(
                        torch.argmax(y_pred, dim=-1) == y_true).item()
                    total += B

                y_pred = model(x_aug1)
                correct += torch.sum(
                    torch.argmax(y_pred, dim=-1) == y_true).item()
                total += B

                loss = loss_fn_classification(y_pred, y_true)

        state_dict['train_loss'] /= total
        state_dict['train_acc'] = correct / total * 100

        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y_true in val_loader:
                if config.contrastive == 'NA':
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
                    B = x.shape[0]
                    assert config.in_channels in [1, 3]
                    if config.in_channels == 1:
                        # Repeat the channel dimension: 1 channel -> 3 channels.
                        x = x.repeat(1, 3, 1, 1)
                    x, y_true = x.to(device), y_true.to(device)

                    y_pred = model(x)

                    # Contrastive loss not computing during validation. Hence putting NaN.
                    state_dict['val_loss'] = torch.nan
                    correct += torch.sum(
                        torch.argmax(y_pred, dim=-1) == y_true).item()
                    total += B

        state_dict['val_loss'] /= total
        state_dict['val_acc'] = correct / total * 100

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        if state_dict['val_acc'] > best_val_acc:
            best_val_acc = state_dict['val_acc']
            best_model = model.state_dict()
            model_save_path = '%s/%s-%s-%s' % (
                config.checkpoint_dir, config.dataset, config.contrastive,
                'best_val_acc.pth')
            torch.save(best_model, model_save_path)
            log('Best model (so far) successfully saved.',
                filepath=log_path,
                to_console=False)

            if state_dict['val_acc'] > 50 and not is_model_saved['val_acc_50%']:
                model_save_path = '%s/%s-%s-%s' % (
                    config.checkpoint_dir, config.dataset, config.contrastive,
                    'val_acc_50%.pth')
                torch.save(best_model, model_save_path)
                is_model_saved['val_acc_50%'] = True
                log('50% accuracy model successfully saved.',
                    filepath=log_path,
                    to_console=False)

            if state_dict['val_acc'] > 70 and not is_model_saved['val_acc_70%']:
                model_save_path = '%s/%s-%s-%s' % (
                    config.checkpoint_dir, config.dataset, config.contrastive,
                    'val_acc_70%.pth')
                torch.save(best_model, model_save_path)
                is_model_saved['val_acc_70%'] = True
                log('70% accuracy model successfully saved.',
                    filepath=log_path,
                    to_console=False)

        if early_stopper.step(state_dict['val_acc']):
            log('Early stopping criterion met. Ending training.',
                filepath=log_path,
                to_console=True)
            break
    return

def infer(config: AttributeHashmap) -> None:
    '''
    Run the model's encoder on the validation set and save the embeddings.
    '''

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    _, val_loader = dataloaders

    model = ResNet50(num_classes=config.num_classes).to(device)

    checkpoint_paths = glob(
        '%s/%s-%s*.pth' %
        (config.checkpoint_dir, config.dataset, config.contrastive))

    for checkpoint in tqdm(checkpoint_paths):
        checkpoint_name = checkpoint.split('/')[-1].replace('.pth', '')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()

        with torch.no_grad():
            for batch_idx, (x, y_true) in enumerate(val_loader):
                B = x.shape[0]
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x = x.repeat(1, 3, 1, 1)
                x, y_true = x.to(device), y_true.to(device)

                z = model.encode(x)

                save_numpy(config=config,
                           batch_idx=batch_idx,
                           checkpoint_name=checkpoint_name,
                           image_batch=x,
                           label_true_batch=y_true,
                           embedding_batch=z)

    return


def save_numpy(config: AttributeHashmap, batch_idx: int, checkpoint_name: str,
               image_batch: torch.Tensor, label_true_batch: torch.Tensor,
               embedding_batch: torch.Tensor):

    image_batch = image_batch.cpu().detach().numpy()
    label_true_batch = label_true_batch.cpu().detach().numpy()
    embedding_batch = embedding_batch.cpu().detach().numpy()
    # channel-first to channel-last
    image_batch = np.moveaxis(image_batch, 1, -1)

    # Save the images, labels, and predictions as numpy files for future reference.
    save_path_numpy = '%s/%s/' % (config.output_save_path, 'embeddings/%s/' %
                                  (checkpoint_name))
    os.makedirs(save_path_numpy, exist_ok=True)

    with open(
            '%s/%s' %
        (save_path_numpy, 'batch_%s.npz' % str(batch_idx).zfill(5)),
            'wb+') as f:
        np.savez(f,
                 image=image_batch,
                 label_true=label_true_batch,
                 embedding=embedding_batch)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--mode', help='Train or infer?', required=True)
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

    assert args.mode in ['train', 'infer']
    if args.mode == 'train':
        train(config=config)
        infer(config=config)
    elif args.mode == 'infer':
        infer(config=config)

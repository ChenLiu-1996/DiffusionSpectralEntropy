# Train some severely overfitting encoders.
# -- quantified by train/val accuracy divergence.
# How to overfit?
# 1. Assign wrong, random labels
# 2. Remove mechanisms to avoid overfitting, e.g., BatchNorm.

import argparse
import os
import sys
from glob import glob
from typing import Tuple

import numpy as np
import torch
import torchvision
import yaml
from tinyimagenet import TinyImageNet
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/nn/')
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from early_stop import EarlyStopping
from log_utils import log
from models import get_model
from path_utils import update_config_dirs
from seed import seed_everything


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
        torchvision_dataset = torchvision.datasets.MNIST
        config.small_image = True

    elif config.dataset == 'cifar10':
        config.in_channels = 3
        config.num_classes = 10
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10
        config.small_image = True

    elif config.dataset == 'cifar100':
        config.in_channels = 3
        config.num_classes = 100
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR100
        config.small_image = True

    elif config.dataset == 'stl10':
        config.in_channels = 3
        config.num_classes = 10
        imsize = 96
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10
        config.small_image = False

    elif config.dataset == 'tinyimagenet':
        config.in_channels = 3
        config.num_classes = 200
        imsize = 64
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet
        config.small_image = False

    elif config.dataset == 'imagenet':
        config.in_channels = 3
        config.num_classes = 1000
        imsize = 224
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet
        config.small_image = False

    else:
        raise ValueError(
            '`config.dataset` value not supported. Value provided: %s.' %
            config.dataset)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(imsize),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=dataset_mean, std=dataset_std)
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

    elif config.dataset in ['stl10']:
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

    if config.bad_method == 'wronglabel':
        train_loader = CorruptLabelDataLoader(train_loader)

    return (train_loader, val_loader), config


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
                                           config.bad_method, config.model,
                                           config.random_seed)

    # Log the config.
    config_str = 'Config: \n'
    for key in config.keys():
        config_str += '%s: %s\n' % (key, config[key])
    config_str += '\nTraining History:'
    log(config_str, filepath=log_path, to_console=False)

    model = get_model(model_name=config.model,
                      num_classes=config.num_classes,
                      small_image=config.small_image).to(device)
    model.init_params()

    opt = torch.optim.AdamW(list(model.encoder.parameters()) +
                            list(model.linear.parameters()),
                            lr=float(config.learning_rate),
                            weight_decay=float(config.weight_decay))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt, T_max=config.max_epoch, eta_min=0)

    loss_fn_classification = torch.nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(mode='max',
                                  patience=config.patience,
                                  percentage=False)

    acc_divergence_pct_list = [20, 30, 40, 50, 60, 70, 80]
    is_model_saved = {}
    for acc_divergence_percentage in acc_divergence_pct_list:
        is_model_saved['acc_divergence_%s%%' %
                       acc_divergence_percentage] = False
    biggest_acc_divergence = 0
    best_model = None

    for epoch_idx in tqdm(range(config.max_epoch)):
        state_dict = {
            'train_loss': 0,
            'train_acc': 0,
            'val_loss': 0,
            'val_acc': 0,
            'divergence': 0,
        }

        #
        '''
        Training
        '''
        model.train()
        correct, total_count = 0, 0
        for _, (x, y_true) in enumerate(train_loader):
            B = x.shape[0]
            assert config.in_channels in [1, 3]
            if config.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model(x)
            loss = loss_fn_classification(y_pred, y_true)
            state_dict['train_loss'] += loss.item() * B
            correct += torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()
            total_count += B

            opt.zero_grad()
            loss.backward()
            opt.step()

        state_dict['train_acc'] = correct / total_count * 100
        state_dict['train_loss'] /= total_count

        if epoch_idx >= 10:
            lr_scheduler.step()

        #
        '''
        Validation (or Linear Probing + Validation)
        '''
        val_loss, val_acc = validate_epoch(
            config=config,
            val_loader=val_loader,
            model=model,
            device=device,
            loss_fn_classification=loss_fn_classification)
        state_dict['val_loss'] = val_loss
        state_dict['val_acc'] = val_acc

        state_dict['divergence'] = \
            state_dict['train_acc'] - state_dict['val_acc']

        log('Epoch: %d. %s' % (epoch_idx, print_state_dict(state_dict)),
            filepath=log_path,
            to_console=False)

        model_save_path = '%s/%s-%s-%s-seed%s-epoch%s-valAcc%.3f%s' % (
            config.checkpoint_dir, config.dataset, config.bad_method,
            config.model, config.random_seed, str(epoch_idx).zfill(4),
            state_dict['val_acc'], '.pth')
        torch.save(model.state_dict(), model_save_path)
        if state_dict['divergence'] > biggest_acc_divergence:
            biggest_acc_divergence = state_dict['divergence']
            best_model = model.state_dict()
            model_save_path = '%s/%s-%s-%s-seed%s-%s' % (
                config.checkpoint_dir, config.dataset, config.bad_method,
                config.model, config.random_seed, 'acc_divergence_biggest.pth')
            torch.save(best_model, model_save_path)
            log('Most train/val divergent model (so far) successfully saved.',
                filepath=log_path,
                to_console=False)

            for acc_divergence_percentage in acc_divergence_pct_list:
                if state_dict['divergence'] > acc_divergence_percentage and \
                        not is_model_saved['acc_divergence_%s%%' % acc_divergence_percentage]:
                    model_save_path = '%s/%s-%s-%s-seed%s-%s' % (
                        config.checkpoint_dir, config.dataset,
                        config.bad_method, config.model, config.random_seed,
                        'acc_divergence_%s%%.pth' % acc_divergence_percentage)
                    torch.save(best_model, model_save_path)
                    is_model_saved['acc_divergence_%s%%' %
                                   acc_divergence_percentage] = True
                    log('%s%% divergence model successfully saved.' %
                        acc_divergence_percentage,
                        filepath=log_path,
                        to_console=False)

        if early_stopper.step(state_dict['divergence']):
            log('Early stopping criterion met. Ending training.',
                filepath=log_path,
                to_console=True)
            break
    return


def validate_epoch(config: AttributeHashmap,
                   val_loader: torch.utils.data.DataLoader,
                   model: torch.nn.Module, device: torch.device,
                   loss_fn_classification: torch.nn.Module):

    correct, total_count = 0, 0
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
            total_count += B

    val_loss /= total_count
    val_acc = correct / total_count * 100

    return val_loss, val_acc


def infer(config: AttributeHashmap) -> None:
    '''
    Run the model's encoder on the validation set and save the embeddings.
    '''

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    dataloaders, config = get_dataloaders(config=config)
    _, val_loader = dataloaders

    model = get_model(model_name=config.model,
                      num_classes=config.num_classes,
                      small_image=config.small_image).to(device)

    checkpoint_paths = sorted(
        glob('%s/%s-%s-%s-%s*.pth' %
             (config.checkpoint_dir, config.dataset, config.bad_method,
              config.model, config.random_seed)))
    log_path = '%s/%s-%s-%s-%s.log' % (config.log_dir, config.dataset,
                                       config.bad_method, config.model,
                                       config.random_seed)

    for checkpoint in tqdm(checkpoint_paths):
        checkpoint_name = checkpoint.split('/')[-1].replace('.pth', '')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()

        total_by_class = {}
        correct_by_class = {}

        with torch.no_grad():
            for batch_idx, (x, y_true) in enumerate(val_loader):
                assert config.in_channels in [1, 3]
                if config.in_channels == 1:
                    # Repeat the channel dimension: 1 channel -> 3 channels.
                    x = x.repeat(1, 3, 1, 1)
                x, y_true = x.to(device), y_true.to(device)

                h = model.encode(x)
                y_pred = model(x)
                y_correct = (torch.argmax(
                    y_pred, dim=-1) == y_true).cpu().detach().numpy()

                # Record per-class accuracy.
                for i in range(y_true.shape[0]):
                    class_true = y_true[i].item()
                    is_correct = y_correct[i]
                    if class_true not in total_by_class.keys():
                        total_by_class[class_true] = 1
                        assert class_true not in correct_by_class.keys()
                        correct_by_class[class_true] = 0
                    else:
                        total_by_class[class_true] += 1
                    if is_correct:
                        correct_by_class[class_true] += 1

                save_numpy(config=config,
                           batch_idx=batch_idx,
                           checkpoint_name=checkpoint_name,
                           image_batch=x,
                           label_true_batch=y_true,
                           embedding_batch=h)

            log('\nCheckpoint: %s' % checkpoint_name,
                filepath=log_path,
                to_console=True)
            log('#total by class:', filepath=log_path, to_console=True)
            log(str(total_by_class), filepath=log_path, to_console=True)
            log('#correct by class:', filepath=log_path, to_console=True)
            log(str(correct_by_class), filepath=log_path, to_console=True)

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
    if args.random_seed is not None:
        config.random_seed = args.random_seed
    config = update_config_dirs(AttributeHashmap(config))

    # Update checkpoint dir.
    config.checkpoint_dir = '%s/%s-%s-%s-seed%s/' % (
        config.checkpoint_dir, config.dataset, config.bad_method, config.model,
        config.random_seed)

    seed_everything(config.random_seed)

    assert args.mode in ['train', 'infer']
    if args.mode == 'train':
        train(config=config)
        infer(config=config)
    elif args.mode == 'infer':
        infer(config=config)

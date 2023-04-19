import argparse
import os
import sys
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
from diffusion_curvature.core import DiffusionMatrix
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tinyimagenet import TinyImageNet
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from seed import seed_everything

sys.path.insert(0, import_dir + '/nn/external_model_loader/')
from barlowtwins_model import BarlowTwinsModel
from characteristics import von_neumann_entropy
from moco_model import MoCoModel
from simsiam_model import SimSiamModel
from swav_model import SwavModel
from vicreg_model import VICRegModel


def compute_diffusion_entropy(embeddings: torch.Tensor, args: AttributeHashmap,
                              eig_npy_path: str) -> float:

    if os.path.exists(eig_npy_path):
        data_numpy = np.load(eig_npy_path)
        eigenvalues_P = data_numpy['eigenvalues_P']
        print('Pre-computed eigenvalues loaded.')
    else:
        # Diffusion Matrix
        diffusion_matrix = DiffusionMatrix(embeddings,
                                           kernel_type="adaptive anisotropic",
                                           k=args.knn)
        print('Diffusion matrix computed.')

        # Diffusion Eigenvalues
        eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
        eigenvalues_P = eigenvalues_P.astype(np.float16)
        print('Eigenvalues computed.')

        with open(eig_npy_path, 'wb+') as f:
            np.savez(f, eigenvalues_P=eigenvalues_P)

    # von Neumann Entropy
    vne = von_neumann_entropy(eigenvalues_P)

    return vne


def get_dataloaders(
    args: AttributeHashmap
) -> Tuple[Tuple[torch.utils.data.DataLoader, ], AttributeHashmap]:

    dataset_dir = '%s/data/%s' % (args.root_dir, args.dataset)

    if args.dataset == 'mnist':
        args.in_channels = 1
        args.num_classes = 10
        imsize = 28
        dataset_mean = (0.1307, )
        dataset_std = (0.3081, )
        torchvision_dataset = torchvision.datasets.MNIST

    elif args.dataset == 'cifar10':
        args.in_channels = 3
        args.num_classes = 10
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR10

    elif args.dataset == 'cifar100':
        args.in_channels = 3
        args.num_classes = 100
        imsize = 32
        dataset_mean = (0.4914, 0.4822, 0.4465)
        dataset_std = (0.2023, 0.1994, 0.2010)
        torchvision_dataset = torchvision.datasets.CIFAR100

    elif args.dataset == 'stl10':
        args.in_channels = 3
        args.num_classes = 10
        imsize = 96
        dataset_mean = (0.4467, 0.4398, 0.4066)
        dataset_std = (0.2603, 0.2566, 0.2713)
        torchvision_dataset = torchvision.datasets.STL10

    elif args.dataset == 'tinyimagenet':
        args.in_channels = 3
        args.num_classes = 200
        imsize = 64
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = TinyImageNet

    elif args.dataset == 'imagenet':
        args.in_channels = 3
        args.num_classes = 1000
        imsize = 224
        dataset_mean = (0.485, 0.456, 0.406)
        dataset_std = (0.229, 0.224, 0.225)
        torchvision_dataset = torchvision.datasets.ImageNet

    if args.in_channels == 3:
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                imsize,
                scale=(0.5, 2.0),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ],
                                               p=0.8),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=dataset_mean,
                                             std=dataset_std)
        ])
    else:
        transform_train = torchvision.transforms.Compose([
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

    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        train_dataset = torchvision_dataset(dataset_dir,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(dataset_dir,
                                          train=False,
                                          download=True,
                                          transform=transform_val)

    elif args.dataset in ['stl10']:
        train_dataset = torchvision_dataset(dataset_dir,
                                            split='train',
                                            download=True,
                                            transform=transform_train)
        val_dataset = torchvision_dataset(dataset_dir,
                                          split='test',
                                          download=True,
                                          transform=transform_val)

    elif args.dataset in ['tinyimagenet', 'imagenet']:
        train_dataset = torchvision_dataset(dataset_dir,
                                            split='train',
                                            transform=transform_train)
        val_dataset = torchvision_dataset(dataset_dir,
                                          split='val',
                                          transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False,
                                             pin_memory=True)

    return train_loader, val_loader


def tune_model(args: AttributeHashmap,
               train_loader: torch.utils.data.DataLoader,
               model: torch.nn.Module, device: torch.device, model_path: str,
               log_path: str):

    # Load the pretrained model.
    model.restore_model()
    model.train()

    if args.full_fine_tune is True:
        model.unfreeze_all()
        if args.learning_rate_tuning is None:
            args.learning_rate_tuning = 1e-4
        opt_probing = torch.optim.AdamW(model.encoder_parameters() +
                                        model.linear_parameters(),
                                        lr=float(args.learning_rate_tuning))
    else:
        model.freeze_all()
        model.unfreeze_linear()
        model.init_linear()
        if args.learning_rate_tuning is None:
            args.learning_rate_tuning = 1e-2
        opt_probing = torch.optim.AdamW(model.linear_parameters(),
                                        lr=float(args.learning_rate_tuning))

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt_probing, T_max=args.num_tuning_epoch // 4)

    best_probing_acc = 0
    for epoch_idx in range(args.num_tuning_epoch):
        probing_acc = tune_model_single_epoch(args=args,
                                              train_loader=train_loader,
                                              model=model,
                                              device=device,
                                              opt_probing=opt_probing)
        lr_scheduler.step()
        log('Probing epoch: %d, acc: %.3f' % (epoch_idx, probing_acc),
            log_path)
        if probing_acc > best_probing_acc:
            best_probing_acc = probing_acc
            model.save_model(model_path)

    model.eval()
    return


def tune_model_single_epoch(args: AttributeHashmap,
                            train_loader: torch.utils.data.DataLoader,
                            model: torch.nn.Module, device: torch.device,
                            opt_probing: torch.optim.Optimizer):

    correct, total_count = 0, 0
    for _, (x, y_true) in enumerate(tqdm(train_loader)):

        B = x.shape[0]
        assert args.in_channels in [1, 3]
        if args.in_channels == 1:
            # Repeat the channel dimension: 1 channel -> 3 channels.
            x = x.repeat(1, 3, 1, 1)
        x, y_true = x.to(device), y_true.to(device)

        y_pred = model.forward(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y_true)
        correct += torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()
        total_count += B

        opt_probing.zero_grad()
        loss.backward()
        opt_probing.step()

    probing_acc = correct / total_count * 100
    model.eval()

    return probing_acc


def infer_model(val_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module, device: torch.device, model_path: str,
                log_path: str) -> float:
    model.restore_model(restore_path=model_path)
    model.eval()
    correct, total_count = 0, 0
    with torch.no_grad():
        for _, (x, y_true) in enumerate(tqdm(val_loader)):
            B = x.shape[0]
            assert args.in_channels in [1, 3]
            if args.in_channels == 1:
                # Repeat the channel dimension: 1 channel -> 3 channels.
                x = x.repeat(1, 3, 1, 1)
            x, y_true = x.to(device), y_true.to(device)

            y_pred = model.forward(x)
            correct += torch.sum(torch.argmax(y_pred, dim=-1) == y_true).item()
            total_count += B

    val_acc_actual = correct / total_count * 100
    log('val acc actual %.2f\n\n' % val_acc_actual, log_path)
    return val_acc_actual


def diffusion_entropy(args: AttributeHashmap):
    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    args.root_dir = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])

    save_folder = './results_diffusion_entropy_PublicModels/%s' % args.dataset
    npy_folder = '%s/%s/' % (save_folder, 'numpy_files')
    pt_folder = '%s/%s/' % (save_folder, 'tuned_models')
    log_path = '%s/log-%s-seed%s-knn%s.txt' % (save_folder, args.dataset,
                                               args.random_seed, args.knn)

    os.makedirs(npy_folder, exist_ok=True)
    os.makedirs(pt_folder, exist_ok=True)

    train_loader, val_loader = get_dataloaders(args=args)

    __models = ['barlowtwins', 'moco', 'simsiam', 'swav', 'vicreg']
    __versions = {
        'barlowtwins': ['barlowtwins_bs2048_ep1000'],
        'moco': ['moco_v1_ep200', 'moco_v2_ep200', 'moco_v2_ep800'],
        'simsiam': ['simsiam_bs256_ep100', 'simsiam_bs512_ep100'],
        'swav': [
            'swav_bs256_ep200',
            'swav_bs256_ep400',
            'swav_bs4096_ep100',
            'swav_bs4096_ep200',
            'swav_bs4096_ep400',
            'swav_bs4096_ep800',
        ],
        'vicreg': ['vicreg_bs2048_ep100']
    }
    top1_acc_nominal = {
        'barlowtwins': [73.5],
        'moco': [60.6, 67.7, 71.1],
        'simsiam': [68.3, 68.1],
        'swav': [72.7, 74.3, 72.1, 73.9, 74.6, 75.3],
        'vicreg': [73.2],
    }
    summary = {}

    for model_name in __models:
        for i, version in enumerate(__versions[model_name]):
            log('model: %s, version: %s' % (model_name, version), log_path)

            summary[version] = {
                'top1_acc_nominal': top1_acc_nominal[model_name][i]
            }

            embedding_npy_path = '%s/%s_%s_embeddings.npy' % (
                npy_folder, version,
                'FineTune' if args.full_fine_tune else 'LinearProbe')
            eig_npy_path = '%s/%s_eigP.npy' % (npy_folder, version)

            if model_name == 'barlowtwins':
                model = BarlowTwinsModel(device=device,
                                         version=version,
                                         num_classes=args.num_classes)
            elif model_name == 'moco':
                model = MoCoModel(device=device,
                                  version=version,
                                  num_classes=args.num_classes)
            elif model_name == 'simsiam':
                model = SimSiamModel(device=device,
                                     version=version,
                                     num_classes=args.num_classes)
            elif model_name == 'swav':
                model = SwavModel(device=device,
                                  version=version,
                                  num_classes=args.num_classes)
            elif model_name == 'vicreg':
                model = VICRegModel(device=device,
                                    version=version,
                                    num_classes=args.num_classes)
            else:
                raise ValueError('model_name: %s not supported.' % model_name)

            #
            '''
            1. Run through encoder and save embeddings.
               We have to do this prior to tuning, because we want to
               see if the manifold characteristics extracted prior to tuning
               can give us enough information on how good the pretraining is.
            '''
            if os.path.exists(embedding_npy_path):
                data_numpy = np.load(embedding_npy_path)
                embeddings = data_numpy['embeddings']
                log('Pre-computed embeddings loaded.', log_path)
            else:
                embeddings = []
                with torch.no_grad():
                    for i, (x, y_true) in enumerate(tqdm(val_loader)):
                        assert args.in_channels in [1, 3]
                        if args.in_channels == 1:
                            # Repeat the channel dimension: 1 channel -> 3 channels.
                            x = x.repeat(1, 3, 1, 1)
                        x, y_true = x.to(device), y_true.to(device)

                        # shape: [B, d, 1, 1]
                        embedding_vec = model.encode(x).cpu().detach().numpy()
                        # shape: [B, d]
                        embedding_vec = embedding_vec.reshape(
                            embedding_vec.shape[:2])
                        embeddings.append(embedding_vec)

                embeddings = np.concatenate(embeddings)
                embeddings = embeddings.astype(np.float16)
                log('Embeddings computed.', log_path)

                with open(embedding_npy_path, 'wb+') as f:
                    np.savez(f, embeddings=embeddings)

            summary[version]['vne'] = compute_diffusion_entropy(
                embeddings=embeddings, args=args, eig_npy_path=eig_npy_path)

            #
            ''' 2. Tune the model. We can either linear probe or full fine tune. '''
            if args.full_fine_tune is True:
                tuned_model_path = '%s/%s_FineTuneModel.pt' % (pt_folder,
                                                               version)
            else:
                tuned_model_path = '%s/%s_LinearProbeModel.pt' % (pt_folder,
                                                                  version)

            if os.path.exists(tuned_model_path):
                log('Loading tuned model: %s' % version, log_path)
                val_acc_actual = infer_model(val_loader=val_loader,
                                             model=model,
                                             device=device,
                                             model_path=tuned_model_path,
                                             log_path=log_path)
            else:
                log('Tuning model: %s ...' % version, log_path)
                if args.full_fine_tune is True:
                    log('NOTE: We are performing a full fine tune!', log_path)
                else:
                    log('NOTE: We are performing a linear probing!', log_path)

                tune_model(args=args,
                           train_loader=train_loader,
                           model=model,
                           device=device,
                           model_path=tuned_model_path,
                           log_path=log_path)
                val_acc_actual = infer_model(val_loader=val_loader,
                                             model=model,
                                             device=device,
                                             model_path=tuned_model_path,
                                             log_path=log_path)
            summary[version]['top1_acc_actual'] = val_acc_actual

    fig_prefix = '%s/diffusion-entropy-PublicModels-%s-%s' % (
        save_folder, args.dataset,
        'FineTune' if args.full_fine_tune else 'LinearProbe')
    plot_summary(summary, fig_prefix=fig_prefix)
    return


def normalize(
        data: Union[np.array, torch.Tensor],
        dynamic_range: List[float] = [0, 1]) -> Union[np.array, torch.Tensor]:
    assert len(dynamic_range) == 2

    x1, x2 = data.min(), data.max()
    y1, y2 = dynamic_range[0], dynamic_range[1]

    slope = (y2 - y1) / (x2 - x1)
    offset = (y1 * x2 - y2 * x1) / (x2 - x1)

    data = data * slope + offset

    # Fix precision issue.
    data = data.clip(y1, y2)
    return data


def plot_summary(summary: dict, fig_prefix: str = None):
    version_list, vne_list, acc_list_nominal, acc_list_actual = [], [], [], []

    fig_vne = plt.figure(figsize=(10, 8))
    ax = fig_vne.add_subplot(1, 1, 1)
    for key in summary.keys():
        version = key
        version_list.append(version)
        vne_list.append(summary[version]['vne'])
        acc_list_nominal.append(summary[version]['top1_acc_nominal'])
        acc_list_actual.append(summary[version]['top1_acc_actual'])

    # This hashmap ensures the same model (different epochs) share the same color.
    model_color_map = {}
    unique_model_names = np.unique(
        [item.split('_ep')[0] for item in version_list])
    for i, name in enumerate(unique_model_names):
        model_color_map[name] = cm.rainbow(i / len(unique_model_names))

    vne_list = np.array(vne_list)
    acc_list_nominal = np.array(acc_list_nominal)
    acc_list_actual = np.array(acc_list_actual)
    scaled_acc_nominal = normalize(
        acc_list_nominal, dynamic_range=[np.min(vne_list),
                                         np.max(vne_list)])
    scaled_acc_actual = normalize(
        acc_list_actual, dynamic_range=[np.min(vne_list),
                                        np.max(vne_list)])

    ax.scatter(version_list, scaled_acc_nominal, color='mediumblue')
    ax.scatter(version_list, scaled_acc_actual, color='firebrick')
    ax.legend(['nominal acc', 'classifier finetune acc'])
    ax.plot(version_list, vne_list, 'k--')
    ax.set_xticks(np.arange(len(version_list)))
    ax.set_xticklabels(version_list, rotation=90)
    ax.spines[['right', 'top']].set_visible(False)

    fig_vne.tight_layout()
    fig_vne.savefig(fig_prefix + '-vne.png')

    fig_corr = plt.figure(figsize=(16, 8))
    ax = fig_corr.add_subplot(1, 2, 1)
    pearson_r, pearson_p = pearsonr(vne_list, acc_list_actual)
    spearman_r, spearman_p = spearmanr(vne_list, acc_list_actual)
    ax.set_title('P.R: %.3f (p = %.3f), S.R: %.3f (p = %.3f)' %
                 (pearson_r, pearson_p, spearman_r, spearman_p))
    ax.set_xlabel('Diffusion Entropy')
    ax.set_ylabel('Linear Probing Accuracy')

    # Plot the performances:
    # same model name: same color
    # different epochs trained : different size
    for i in range(len(version_list)):
        ax.scatter(vne_list[i],
                   acc_list_actual[i],
                   color=model_color_map[version_list[i].split('_ep')[0]],
                   s=int(version_list[i].split('_ep')[1]) / 5,
                   cmap='tab20')
    ax.legend(version_list)
    ax.spines[['right', 'top']].set_visible(False)

    ax = fig_corr.add_subplot(1, 2, 2)
    pearson_r, pearson_p = pearsonr(acc_list_nominal, acc_list_actual)
    spearman_r, spearman_p = spearmanr(acc_list_nominal, acc_list_actual)
    ax.set_title('P.R: %.3f (p = %.3f), S.R: %.3f (p = %.3f)' %
                 (pearson_r, pearson_p, spearman_r, spearman_p))
    ax.set_xlabel('Nominal Accuracy')
    ax.set_ylabel('Linear Probing Accuracy')

    for i in range(len(version_list)):
        ax.scatter(acc_list_nominal[i],
                   acc_list_actual[i],
                   color=model_color_map[version_list[i].split('_ep')[0]],
                   s=int(version_list[i].split('_ep')[1]) / 5,
                   cmap='tab20')
    ax.legend(version_list)
    ax.spines[['right', 'top']].set_visible(False)

    fig_corr.tight_layout()
    fig_corr.savefig(fig_prefix + '-correlation.png')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id',
                        help='which gpu to use',
                        type=int,
                        default=0)
    parser.add_argument('--dataset',
                        help='which dataset to run',
                        type=str,
                        default='mnist')
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    parser.add_argument('--random-seed',
                        help='random seed.',
                        type=int,
                        default=0)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--full-fine-tune',
                        help='If True, full fine tune. Else, linear probe.',
                        action='store_true')
    parser.add_argument('--learning_rate_tuning', type=float, default=None)
    parser.add_argument('--num_tuning_epoch', type=int, default=100)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    seed_everything(args.random_seed)

    diffusion_entropy(args=args)

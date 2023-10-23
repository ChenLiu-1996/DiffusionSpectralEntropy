import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import random
import torch
from sklearn.datasets import make_blobs

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/api/')
from dsmi import diffusion_spectral_mutual_information, adjacency_spectral_mutual_information

sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap

sys.path.insert(0, import_dir + '/src/comparison/MINE/')
from mine.models.mine import MutualInformationEstimator
from pytorch_lightning import Trainer

# sys.path.insert(0, import_dir + '/src/comparison/NPEET_LNC/')
# from lnc import MI as MI_LNC

sys.path.insert(0, import_dir + '/src/comparison/NPEET/')
from npeet import entropy_estimators as MI_npeet

# sys.path.insert(0, import_dir + '/src/comparison/EDGE/')
# from EDGE_4_4_1 import EDGE as MI_EDGE


def test_MINE(model: MutualInformationEstimator,
              test_dataloader: torch.utils.data.Dataset,
              device: torch.device) -> float:
    model = model.to(device)
    model.eval()
    mis = []

    for _, (x, z) in enumerate(test_dataloader):
        x = x.to(device)
        z = z.to(device)

        loss = model.energy_loss(x, z)
        mi = -loss
        mis.append(mi.item())

    avg_mi = np.mean(mis)

    return avg_mi


def train_MINE(dimX: int, dimY: int, lr: float, batch_size: int, epochs: int,
               train_loader: torch.utils.data.Dataset,
               device: torch.device) -> MutualInformationEstimator:
    kwargs = {
        'lr': lr,
        'batch_size': batch_size,
        'train_loader': train_loader,
        'test_loader': train_loader,  # placeholder
        'alpha': 1.0,
    }

    model = MutualInformationEstimator(dimX,
                                       dimY,
                                       loss='mine_biased',
                                       **kwargs).to(device)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model)

    return model


class ZipTwoDataset(torch.utils.data.Dataset):

    def __init__(self, X: np.array, Y: np.array):
        assert X.shape[0] == Y.shape[0]

        self.N = X.shape[0]
        X = X.reshape(self.N, -1)
        Y = Y.reshape(self.N, -1)

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __getitem__(self, idx):
        a, b = self.X[idx, :], self.Y[idx, :]
        return a, b

    def __len__(self):
        return self.N


def corrupt_label(labels: np.array,
                  corruption_ratio: float,
                  random_seed: int = 1):
    assert corruption_ratio >= 0 and corruption_ratio <= 1
    if corruption_ratio == 0:
        return labels
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        indices = random.sample(range(len(labels)),
                                k=int(corruption_ratio * len(labels)))
        permuted_indices = np.random.permutation(indices)
        corrupted_labels = labels.copy()
        corrupted_labels[indices] = corrupted_labels[permuted_indices]
        return corrupted_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/ASMI-toy-MI-blob.png' % (save_root)

    device = torch.device('cuda:%s' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    dim_list = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    corruption_ratio_list = np.linspace(0.0, 1.0, 11)
    noise_level_list = [1e-2, 1e-1, 5e-1, 7e-1]
    num_repetition = 2

    method_list = [
        'DSMI', 'DMEE', 'ASMI_KNN', 'ASMI_Gaussian', 'ASMI_Anisotropic'
    ]

    mi_by_corruption_d_20_dict = {}
    mi_by_corruption_d_100_dict = {}
    mi_by_dim_dict = {}

    for m in method_list:
        mi_by_corruption_d_20_dict[m] = [[
            [] for _ in range(len(noise_level_list))
        ] for _ in range(len(corruption_ratio_list))]
        mi_by_corruption_d_100_dict[m] = [[
            [] for _ in range(len(noise_level_list))
        ] for _ in range(len(corruption_ratio_list))]
        mi_by_dim_dict[m] = [[[] for _ in range(len(noise_level_list))]
                             for _ in range(len(dim_list))]

    # Experiment 1: vary the label corruption.
    default_dim = 20
    for i, corruption_ratio in enumerate(tqdm(corruption_ratio_list)):
        blobs_data, blobs_label = make_blobs(n_samples=5000,
                                             n_features=default_dim)
        blobs_label = corrupt_label(blobs_label,
                                    corruption_ratio=corruption_ratio)
        for j, noise_level in enumerate(noise_level_list):
            blobs_data += noise_level * np.random.uniform(
                -1, 1, size=blobs_data.shape)
            for rep in range(num_repetition):

                mi_by_corruption_d_20_dict['DSMI'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_corruption_d_20_dict['DMEE'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        matrix_entry_entropy=True,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_corruption_d_20_dict['ASMI_KNN'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        use_knn=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])
                
                mi_by_corruption_d_20_dict['ASMI_Gaussian'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])
                
                mi_by_corruption_d_20_dict['ASMI_Anisotropic'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        anisotropic=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])

    default_dim = 100
    for i, corruption_ratio in enumerate(tqdm(corruption_ratio_list)):
        blobs_data, blobs_label = make_blobs(n_samples=5000,
                                             n_features=default_dim)
        blobs_label = corrupt_label(blobs_label,
                                    corruption_ratio=corruption_ratio)
        for j, noise_level in enumerate(noise_level_list):
            blobs_data += noise_level * np.random.uniform(
                -1, 1, size=blobs_data.shape)
            for rep in range(num_repetition):

                mi_by_corruption_d_100_dict['DSMI'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_corruption_d_100_dict['DMEE'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        matrix_entry_entropy=True,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_corruption_d_100_dict['ASMI_KNN'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        use_knn=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])
                
                mi_by_corruption_d_100_dict['ASMI_Gaussian'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])
                
                mi_by_corruption_d_100_dict['ASMI_Anisotropic'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        anisotropic=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(default_dim),
                        random_seed=rep)[0])


    # Experiment 2: vary the dimension.
    for i, dim in enumerate(tqdm(dim_list)):
        blobs_data, blobs_label = make_blobs(n_samples=5000, n_features=dim)
        for j, noise_level in enumerate(noise_level_list):
            blobs_data += noise_level * np.random.uniform(
                -1, 1, size=blobs_data.shape)
            for rep in range(num_repetition):

                mi_by_dim_dict['DSMI'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_dim_dict['DMEE'][i][j].append(
                    diffusion_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        matrix_entry_entropy=True,
                        reference_discrete=True,
                        t=1,
                        gaussian_kernel_sigma=np.sqrt(dim),
                        chebyshev_approx=False,
                        random_seed=rep)[0])
                
                mi_by_dim_dict['ASMI_KNN'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        use_knn=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(dim),
                        random_seed=rep)[0])
                
                mi_by_dim_dict['ASMI_Gaussian'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(dim),
                        random_seed=rep)[0])
                
                mi_by_dim_dict['ASMI_Anisotropic'][i][j].append(
                    adjacency_spectral_mutual_information(
                        embedding_vectors=blobs_data,
                        reference_vectors=blobs_label,
                        anisotropic=True,
                        reference_discrete=True,
                        gaussian_kernel_sigma=np.sqrt(dim),
                        random_seed=rep)[0])

    for m in method_list:
        mi_by_corruption_d_20_dict[m] = np.array(mi_by_corruption_d_20_dict[m])
        mi_by_corruption_d_100_dict[m] = np.array(
            mi_by_corruption_d_100_dict[m])
        mi_by_dim_dict[m] = np.array(mi_by_dim_dict[m])

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 10
    fig_mi = plt.figure(figsize=(25, 6))
    ax = fig_mi.add_subplot(1, 3, 1)
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashed', 'dotted']

    for m, method in enumerate(method_list):
        for j in range(len(noise_level_list)):
            ax.plot(corruption_ratio_list,
                    np.mean(mi_by_corruption_d_20_dict[method][:, j, :],
                            axis=-1),
                    color=cm.get_cmap('tab10').colors[m],
                    marker='o',
                    alpha=0.5,
                    linestyle=linestyle_list[j])

            ax.fill_between(
                corruption_ratio_list,
                np.mean(mi_by_corruption_d_20_dict[method][:, j, :], axis=-1) -
                np.std(mi_by_corruption_d_20_dict[method][:, j, :], axis=-1),
                np.mean(mi_by_corruption_d_20_dict[method][:, j, :], axis=-1) +
                np.std(mi_by_corruption_d_20_dict[method][:, j, :], axis=-1),
                color=cm.get_cmap('tab10').colors[m],
                alpha=0.2,
                label='_nolegend_')

    ax.legend([
        r'%s, |noise| = %d%%' % (m, noise * 100) for m in method_list
        for noise in noise_level_list
    ],
              ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Label Corruption Ratio (with $d$ = 20)', fontsize=16)
    ax.set_ylabel('Estimated Mutual Information', fontsize=16)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=bottom, top=top+1)

    ax = fig_mi.add_subplot(1, 3, 2)
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashed', 'dotted']

    for m, method in enumerate(method_list):
        for j in range(len(noise_level_list)):
            ax.plot(corruption_ratio_list,
                    np.mean(mi_by_corruption_d_100_dict[method][:, j, :],
                            axis=-1),
                    color=cm.get_cmap('tab10').colors[m],
                    marker='o',
                    alpha=0.5,
                    linestyle=linestyle_list[j])

            ax.fill_between(
                corruption_ratio_list,
                np.mean(mi_by_corruption_d_100_dict[method][:, j, :],
                        axis=-1) -
                np.std(mi_by_corruption_d_100_dict[method][:, j, :], axis=-1),
                np.mean(mi_by_corruption_d_100_dict[method][:, j, :],
                        axis=-1) +
                np.std(mi_by_corruption_d_100_dict[method][:, j, :], axis=-1),
                color=cm.get_cmap('tab10').colors[m],
                alpha=0.2,
                label='_nolegend_')

    ax.legend([
        r'%s, |noise| = %d%%' % (m, noise * 100) for m in method_list
        for noise in noise_level_list
    ],
              ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Label Corruption Ratio (with $d$ = 100)', fontsize=16)
    ax.set_ylabel('Estimated Mutual Information', fontsize=16)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=bottom, top=top+1)

    ax = fig_mi.add_subplot(1, 3, 3)
    ax.spines[['right', 'top']].set_visible(False)
    linestyle_list = ['solid', 'dashed', 'dotted']

    for m, method in enumerate(method_list):
        for j in range(len(noise_level_list)):
            ax.plot(np.log10(dim_list),
                    np.mean(mi_by_dim_dict[method][:, j, :], axis=-1),
                    color=cm.get_cmap('tab10').colors[m],
                    marker='o',
                    alpha=0.5,
                    linestyle=linestyle_list[j])

            ax.fill_between(np.log10(dim_list),
                            np.mean(mi_by_dim_dict[method][:, j, :], axis=-1) -
                            np.std(mi_by_dim_dict[method][:, j, :], axis=-1),
                            np.mean(mi_by_dim_dict[method][:, j, :], axis=-1) +
                            np.std(mi_by_dim_dict[method][:, j, :], axis=-1),
                            color=cm.get_cmap('tab10').colors[m],
                            alpha=0.2,
                            label='_nolegend_')

    ax.legend([
        r'%s, |noise| = %d%%' % (m, noise * 100) for m in method_list
        for noise in noise_level_list
    ],
              ncol=2)
    ax.set_xticks(np.log10(dim_list))
    ax.set_xticklabels(dim_list, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Dimension $D$ (log scale)', fontsize=16)
    ax.set_ylabel('Estimated Mutual Information', fontsize=16)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=bottom, top=top+1)

    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig)
    plt.close(fig=fig_mi)

import argparse
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
import random
import torch

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

sys.path.append('../../api/')
from dsmi import diffusion_spectral_mutual_information
sys.path.append('../../src/utils/')
from attribute_hashmap import AttributeHashmap

from mine.models.mine import MutualInformationEstimator
from pytorch_lightning import Trainer

def test_MINE(model: MutualInformationEstimator, 
              test_dataloader: torch.utils.data.Dataset,
              device: torch.device) -> float:
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


def train_MINE(dimX: int,
               dimY: int,
               lr: float, 
               batch_size: int, 
               epochs: int, 
               train_loader: torch.utils.data.Dataset, 
               device: torch.device) -> MutualInformationEstimator:
    kwargs = {
        'lr': lr,
        'batch_size': batch_size,
        'train_loader': train_loader,
        'test_loader': train_loader, # placeholder
        'alpha': 1.0
    }
    
    model = MutualInformationEstimator(
                dimX, dimY, loss='mine_biased', **kwargs).to(device)
            
    trainer = Trainer(max_epochs=epochs)
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


def generate_tree(dim: int = 2,
                  num_branches: int = 5,
                  num_points: int = 1000,
                  random_seed: int = 1):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Define a function to generate a branch
    def generate_branch(start_point, length, dim):
        # Define a random direction for the branch
        direction = np.random.uniform(low=-1, high=1, size=dim)

        # Generate a set of points along the branch
        branch_points = [start_point]
        for _ in range(int(length)):
            branch_points.append(branch_points[-1] + direction * 0.1)
            direction += np.random.normal(size=dim) * 0.05
            direction /= np.linalg.norm(direction)

        return branch_points

    # Generate the root point
    root_point = np.random.normal(size=dim)

    # Generate the branches
    branches = []
    classes = []
    cumulative_length = 0
    for i in range(num_branches):
        # Generate a random start point for the branch
        start_point = root_point + np.random.normal(size=dim) * 0.1

        # Generate a random length and angle for the branch
        if i < num_branches - 1:
            length = np.random.randint(num_points // (num_branches + 1),
                                       num_points // num_branches)
            cumulative_length += length + 1
        else:
            length = num_points - cumulative_length - 1

        # Generate the points for the branch
        branch_points = generate_branch(start_point, length, dim)

        # Add the branch points to the list of all points
        branches.append(branch_points)
        classes += [i] * (length + 1)

    # Concatenate all the branch points into a single array
    points = np.concatenate(branches)
    classes = np.array(classes)

    # Add some noise to the points
    noise = np.random.normal(size=(points.shape[0], dim)) * 0.2
    points += noise

    return points, classes


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
    parser.add_argument('--gaussian-kernel-sigma', type=float, default=10.0)
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    save_path_fig = '%s/toy-data-MI.png' % (save_root)

    device = torch.device(
        'cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 14

    D = 30
    num_corruption_ratio = 20
    num_repetition = 1

    num_branch_list = [5, 10, 20]  # Currently hard-coded. Has to be 3 items.
    t_list = [1, 2, 3, 5]
    noise_level_list = [1e-2, 1e-1, 5e-1]

    corruption_ratio_list = np.linspace(0, 1, num_corruption_ratio)
    mi_Y_sample_list_tree = [[[[[] for _ in range(num_repetition)]
                               for _ in range(len(t_list))]
                              for _ in range(len(noise_level_list))]
                             for _ in range(len(num_branch_list))]
    mi_Y_shannon_list_tree = [[[[] for _ in range(num_repetition)]
                               for _ in range(len(noise_level_list))]
                              for _ in range(len(num_branch_list))]
    mi_Y_MINE_list_tree = [[[[] for _ in range(num_repetition)]
                               for _ in range(len(noise_level_list))]
                              for _ in range(len(num_branch_list))]

    for i in range(num_repetition):
        for corruption_ratio in tqdm(corruption_ratio_list):
            for j, t in enumerate(t_list):
                for k, noise_level in enumerate(noise_level_list):
                    for b, num_branches in enumerate(num_branch_list):

                        tree_data, tree_clusters = generate_tree(
                            dim=D,
                            num_branches=num_branches,
                            num_points=100 * num_branches,
                            random_seed=0)
                        tree_data += noise_level * np.random.uniform(
                            -1, 1, size=tree_data.shape)

                        tree_clusters = corrupt_label(
                            tree_clusters, corruption_ratio=corruption_ratio)
                        
                        mi_Y_sample, _ = diffusion_spectral_mutual_information(
                            embedding_vectors=tree_data,
                            reference_vectors=tree_clusters,
                            reference_discrete=True,
                            t=t,
                            gaussian_kernel_sigma=args.gaussian_kernel_sigma,
                            chebyshev_approx=False)

                        mi_Y_sample_list_tree[b][k][j][i].append(mi_Y_sample)

                        if j == 0:
                            # Compute CSMI only once.
                            mi_Y_shannon, _ = diffusion_spectral_mutual_information(
                                embedding_vectors=tree_data,
                                reference_vectors=tree_clusters,
                                classic_shannon_entropy=True,
                                t=t,
                                gaussian_kernel_sigma=args.gaussian_kernel_sigma,
                                chebyshev_approx=False)
                            mi_Y_shannon_list_tree[b][k][i].append(
                                mi_Y_shannon)
                            
                            # Compute MINE only once.
                            bs = 500
                            mine_loader = torch.utils.data.DataLoader(ZipTwoDataset(tree_data, tree_clusters),
                                                                      batch_size=bs, shuffle=True)
                            model = train_MINE(dimX=D,dimY=1,lr=1e-4,batch_size=bs,epochs=200,
                                               train_loader=mine_loader,
                                               device=device)
                            
                            mine_mi = test_MINE(model=model, test_dataloader=mine_loader, device=device)
                            mi_Y_MINE_list_tree[b][k][i].append(mine_mi)


    mi_Y_sample_list_tree = np.array(mi_Y_sample_list_tree)
    mi_Y_shannon_list_tree = np.array(mi_Y_shannon_list_tree)
    mi_Y_MINE_list_tree = np.array(mi_Y_MINE_list_tree)

    fig_mi = plt.figure(figsize=(32, 10))
    gs = GridSpec(5, 9, figure=fig_mi)

    for num_branch_idx, corruption_ratio, gs_y in zip(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0.5, 1.0, 0, 0.5, 1.0, 0, 0.5, 1.0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8]):
        ax = fig_mi.add_subplot(gs[0, gs_y], projection='3d')
        ax.spines[['right', 'top']].set_visible(False)

        # Tree.
        tree_data, tree_clusters = generate_tree(
            dim=3,
            num_branches=num_branch_list[num_branch_idx],
            num_points=100 * num_branch_list[num_branch_idx],
            random_seed=0)
        tree_clusters = corrupt_label(tree_clusters,
                                      corruption_ratio=corruption_ratio)
        ax.scatter(tree_data[:, 0],
                   tree_data[:, 1],
                   tree_data[:, 2],
                   c=np.array(cm.get_cmap('tab20').colors)[tree_clusters],
                   alpha=1)

    for num_branch_idx, gs_y_begin in zip([0, 1, 2], [0, 3, 6]):
        ax = fig_mi.add_subplot(gs[1:3, gs_y_begin:gs_y_begin + 3])
        ax.spines[['right', 'top']].set_visible(False)
        linestyle_list = ['solid', 'dashed', 'dotted']
        for j in range(len(t_list)):
            for k in range(len(noise_level_list)):
                ax.plot(corruption_ratio_list,
                        np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j,
                                                      ...],
                                axis=0),
                        color=cm.get_cmap('tab10').colors[j],
                        linestyle=linestyle_list[k])
        ax.legend([
            r'$t$ = %d, |noise| = %d%%' % (t, noise * 100) for t in t_list
            for noise in noise_level_list
        ],
                  loc='upper right',
                  ncol=2)
        for j in range(len(t_list)):
            for k in range(len(noise_level_list)):
                ax.fill_between(
                    corruption_ratio_list,
                    np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                            axis=0) -
                    np.std(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                           axis=0),
                    np.mean(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                            axis=0) +
                    np.std(mi_Y_sample_list_tree[num_branch_idx, k, j, ...],
                           axis=0),
                    color=cm.get_cmap('tab10').colors[j],
                    alpha=0.2)
        ax.axhline(y=0, color='gray', linestyle='-.')
        ax.tick_params(axis='both', which='major', labelsize=20)
        if num_branch_idx == 0:
            ax.set_ylabel('DSMI', fontsize=20)

        # Plot CSMI.
        ax = fig_mi.add_subplot(gs[3:4, gs_y_begin:gs_y_begin + 3])
        ax.spines[['right', 'top']].set_visible(False)
        linestyle_list = ['solid', 'dashed', 'dotted']
        for k in range(len(noise_level_list)):
            ax.plot(corruption_ratio_list,
                    np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                            axis=0),
                    color=cm.get_cmap('tab10').colors[0],
                    linestyle=linestyle_list[k])
        ax.legend(
            [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
            loc='upper right',
            ncol=3)
        for k in range(len(noise_level_list)):
            ax.fill_between(
                corruption_ratio_list,
                np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                        axis=0) -
                np.std(mi_Y_shannon_list_tree[num_branch_idx, k, ...], axis=0),
                np.mean(mi_Y_shannon_list_tree[num_branch_idx, k, ...],
                        axis=0) +
                np.std(mi_Y_shannon_list_tree[num_branch_idx, k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[0],
                alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Label Corruption Ratio', fontsize=25)
        if num_branch_idx == 0:
            ax.set_ylabel('CSMI', fontsize=20)
        
        # Plot MINE.
        ax = fig_mi.add_subplot(gs[4:, gs_y_begin:gs_y_begin + 3])
        ax.spines[['right', 'top']].set_visible(False)
        linestyle_list = ['solid', 'dashed', 'dotted']
        for k in range(len(noise_level_list)):
            ax.plot(corruption_ratio_list,
                    np.mean(mi_Y_MINE_list_tree[num_branch_idx, k, ...],
                            axis=0),
                    color=cm.get_cmap('tab10').colors[0],
                    linestyle=linestyle_list[k])
        ax.legend(
            [r'|noise| = %d%%' % (noise * 100) for noise in noise_level_list],
            loc='upper right',
            ncol=3)
        for k in range(len(noise_level_list)):
            ax.fill_between(
                corruption_ratio_list,
                np.mean(mi_Y_MINE_list_tree[num_branch_idx, k, ...],
                        axis=0) -
                np.std(mi_Y_MINE_list_tree[num_branch_idx, k, ...], axis=0),
                np.mean(mi_Y_MINE_list_tree[num_branch_idx, k, ...],
                        axis=0) +
                np.std(mi_Y_MINE_list_tree[num_branch_idx, k, ...], axis=0),
                color=cm.get_cmap('tab10').colors[0],
                alpha=0.2)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('Label Corruption Ratio', fontsize=25)
        if num_branch_idx == 0:
            ax.set_ylabel('MINE', fontsize=20)



    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig)
    plt.close(fig=fig_mi)

import numpy as np
import torch
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt

from mine.models.mine import MutualInformationEstimator
from mine.datasets import MultivariateNormalDataset, BivariateNormalDataset

import sys
sys.path.append('../../api/')
from dsmi import diffusion_spectral_mutual_information

import logging
logging.getLogger().setLevel(logging.ERROR)

def test_DSMI(test_dataloader: torch.utils.data.Dataset):
    # TODO: batch or not?
    X = []
    Z = []

    for _, (x, z) in enumerate(test_dataloader):
        X.append(x.cpu().detach().numpy()) # (batch_size, dim)
        Z.append(z.cpu().detach().numpy()) # (batch_size, dim)
    
    X = np.concatenate(X, axis=0) # (N, dim)
    Z = np.concatenate(Z, axis=0) # (N, dim)
    
    print('X.shape, Z.shape: ', X.shape, Z.shape)
    print('X.min, X.max: ', X.min(), X.max(), 'Z.min, Z.max: ', Z.min(), Z.max())
    print('X.mean, X.std: ', X.mean(), X.std(), 'Z.mean, Z.std: ', Z.mean(), Z.std())

    dsmi, _ = diffusion_spectral_mutual_information(
        embedding_vectors=Z,
        reference_vectors=X,
        gaussian_kernel_sigma=1,
        n_clusters=5, # TODO?
        precomputed_clusters=None)
    
    return dsmi
    

def test_MINE(model: MutualInformationEstimator, 
              test_dataloader: torch.utils.data.Dataset,
              device: torch.device):
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


def train_MINE(device):
    dimX = 10
    dimY = 10
    N = 3000
    lr = 1e-4
    epochs = 200
    batch_size = 500

    steps = 15
    rhos = np.linspace(-0.99, 0.99, steps)
    loss_type = ['mine_biased']

    results_dict = dict()

    for loss in loss_type:
        results = []
        for rho in rhos:
            train_loader = torch.utils.data.DataLoader(
                MultivariateNormalDataset(N, dimX, rho), batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                MultivariateNormalDataset(N, dimX, rho), batch_size=batch_size, shuffle=True)

            
            true_mi = test_loader.dataset.true_mi
            true_entropy = test_loader.dataset.true_entropy
            true_X_entropy = test_loader.dataset.true_X_entropy

            kwargs = {
                'lr': lr,
                'batch_size': batch_size,
                'train_loader': train_loader,
                'test_loader': test_loader,
                'alpha': 1.0
            }

            model = MutualInformationEstimator(
                dimX, dimY, loss=loss, **kwargs).to(device)
            
            trainer = Trainer(max_epochs=epochs)
            trainer.fit(model)
            
            avg_test_mi = test_MINE(model, test_loader, device)
            dsmi = test_DSMI(test_loader)

            print("True_mi {}, True Entropy {}, True X Entropy {}".format(
                true_mi, true_entropy, true_X_entropy))
            print("MINE {}".format(avg_test_mi))
            print("DSMI {}".format(dsmi))
            results.append((rho, avg_test_mi, true_mi, dsmi))

        results = np.array(results)
        results_dict[loss] = results

        fig, axs = plt.subplots(1, len(loss_type), sharex = True, figsize = (6,4))
        plots = []
        for ix, loss in enumerate(loss_type):
            results = results_dict[loss]
            plots += axs.plot(results[:,0], results[:,1], label='MINE', color='red')
            plots += axs.plot(results[:,0], results[:,3], label='DSMI', color='blue')
            plots += axs.plot(results[:,0], results[:,2], linestyle='--', label='True MI', color='black')
            axs.set_xlabel('correlation')
            axs.set_ylabel('mi')
            #axs.title.set_text(f"{loss} for {dim} dimensional inputs")
            
        fig.legend(plots[0:2], labels = ['MINE', 'True MI'], loc='upper right')
        fig.savefig('figures/mi_estimation.png')
    


if __name__ == '__main__':
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    train_MINE(device)
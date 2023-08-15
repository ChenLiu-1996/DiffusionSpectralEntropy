import numpy as np
import torch


class DSE_Loss(torch.nn.Module):
    '''
    Diffusion Spectral Entropy as a loss.

    DSE over a set of N vectors, each of D dimensions.

    DSE = - sum_i [eig_i^t log eig_i^t]

    where each `eig_i` is an eigenvalue of `P`,
    where `P` is the diffusion matrix computed on the data graph of the [N, D] vectors.

    In theory, this can be applied to the outputs of any layer in a neural network.
    For now, let's assume it's applied to the penultimate layer of a classification model.
    '''

    def __init__(self,
                 sigma: int = 10,
                 t: int = 1,
                 eps: float = 1e-6,
                 min_samples: int = 1000) -> None:
        '''
        sigma:
            conceptually, the neighborhood size of Gaussian kernel.
        t:
            power of the diffusion eigenvalue prior to entropy computation.
        eps:
            small value for numerical stability in `log`.
        min_samples:
            minimum number of data points accumulated before computation.
        '''
        super().__init__()
        self.sigma = sigma
        self.t = t
        self.eps = eps
        self.min_samples = min_samples

        self.embedding_vectors = None

    def forward(self, x) -> torch.Tensor:
        assert len(x.shape) == 2, \
        'DSE_Loss currently only supports tensors with 2 dimensions.'

        # Accumulate embedding vectors until enough samples are gathered.
        if self.embedding_vectors is None or self.embedding_vectors.shape[
                0] < self.min_samples:
            if self.embedding_vectors is None:
                self.embedding_vectors = x
            else:
                self.embedding_vectors = torch.cat([self.embedding_vectors, x],
                                                   dim=0)
            # Backprop nothing until enough samples are gathered.
            return 0

        # Diffusion matrix
        K = diffusion_matrix_with_gradient(x)
        eigenvalues = torch.linalg.eigvalsh(K)

        # Eigenvalues may be negative. Only care about the magnitude, not the sign.
        eigenvalues = torch.abs(eigenvalues)

        # Power eigenvalues to `t` to mitigate effect of noise.
        eigenvalues = eigenvalues**self.t

        prob = eigenvalues / eigenvalues.sum()
        prob = prob + self.eps

        DSE = -torch.sum(prob * torch.log2(prob))

        return DSE


def diffusion_matrix_with_gradient(X: torch.Tensor,
                                   sigma: float = 10.0) -> torch.Tensor:
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X
    Returns a diffusion matrix P.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        K: a numpy array of size n x n that has the same eigenvalues as the diffusion matrix.
    '''

    # Construct the distance matrix.
    D = torch.cdist(X, X)

    # Gaussian kernel
    G = (1 / (sigma * np.sqrt(2 * np.pi))) * torch.exp(
        (-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    Deg = torch.diag(1 / torch.sum(G, axis=1)**0.5)
    K = Deg @ G @ Deg

    # Now K has the exact same eigenvalues as the diffusion matrix `P`
    # which is defined as `P = D^{-1} K`, with `D = np.diag(np.sum(K, axis=1))`.

    return K

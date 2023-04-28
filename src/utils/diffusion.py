import numpy as np
from sklearn.metrics import pairwise_distances


def compute_diffusion_matrix(X: np.array,
                             k: int = 20,
                             eps: float = 1e-6,
                             density_norm_pow: float = 0.5):
    """
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input12 X returns a diffusion matrix P, as an numpy ndarray.
    X is a numpy array of size n x d
    Using "adaptive anisotropic" kernel
    k is the adaptive kernel parameter
    Returns:
    P is a numpy array of size n x n that is the diffusion matrix
    """
    # construct the distance matrix
    D = pairwise_distances(X)
    # make the affinity matrix

    # In case N <= K
    assert X.shape[0] > 1
    k = min(k, X.shape[0] - 1)

    # Get the distance to the k-th neighbor
    distance_to_k_neighbor = np.partition(D, k)[:, k]

    # Populate matrices with this distance for easy division.
    div1 = np.ones(len(D))[:, None] @ distance_to_k_neighbor[None, :]
    div2 = distance_to_k_neighbor[:, None] @ np.ones(len(D))[None, :]

    # Compute the gaussian kernel with an adaptive bandwidth
    W = (1 / (2 * np.sqrt(2 * np.pi))) * (np.exp(-D**2 /
                                                 (2 * div1**2)) / div1 +
                                          np.exp(-D**2 / (2 * div2**2)) / div2)

    # Anisotropic density normalization.
    if density_norm_pow > 0:
        D = np.diag(1 / np.sum(W, axis=1)**density_norm_pow)
        W = D @ W @ D

    # Turn affinity matrix into diffusion matrix
    W = W + np.eye(len(X)) * eps
    D = np.diag(1 / np.sum(W, axis=1))
    P = D @ W

    return P

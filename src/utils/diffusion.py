import numpy as np
from sklearn.metrics import pairwise_distances


def compute_diffusion_matrix(X: np.array,
                             k: int = 10,
                             density_norm_pow: float = 1.0):
    """
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using "adaptive anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        k: k-nearest-neighbor parameter
        density_norm_pow: a float in [0, 1]
            == 0: classic Gaussian kernel
            == 1: completely removes density and provides a geometric equivalent to
                  uniform sampling of the underlying manifold
    Returns:
        P: a numpy array of size n x n that is the diffusion matrix
    """
    # Construct the distance matrix.
    D = pairwise_distances(X)

    # In case N <= K
    assert X.shape[0] > 1
    k = min(k, X.shape[0] - 1)

    # Get the distance to the k-th neighbor.
    distance_to_k_neighbor = np.partition(D, k)[:, k]

    # Populate matrices with this distance for easy division.
    div1 = np.ones(len(D))[:, None] @ distance_to_k_neighbor[None, :]
    div2 = distance_to_k_neighbor[:, None] @ np.ones(len(D))[None, :]

    # Compute the gaussian kernel with an adaptive bandwidth
    W = (1 / np.sqrt(2 * np.pi)) * (np.exp(-D**2 / (2 * div1**2)) / div1 +
                                    np.exp(-D**2 / (2 * div2**2)) / div2)

    # Anisotropic density normalization.
    if density_norm_pow > 0:
        Deg = np.diag(1 / np.sum(W, axis=1)**density_norm_pow)
        W = Deg @ W @ Deg

    # Turn affinity matrix into diffusion matrix.
    Deg = np.diag(1 / np.sum(W, axis=1))
    P = Deg @ W

    return P

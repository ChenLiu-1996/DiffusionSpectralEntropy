import numpy as np
from sklearn.metrics import pairwise_distances
import phate
# import magic
import warnings

warnings.filterwarnings("ignore")

# import graphtools
# def compute_diffusion_matrix(X, k):
#     graph = graphtools.Graph(
#         X,
#         knn=k,
#         decay=1,
#         thresh=1e-4,
#         verbose=False,
#         random_state=1,
#     )
#     import pdb
#     pdb.set_trace()
#     return graph.diff_op.toarray()


def diffusion_matrix_from_phate_distance(X: np.array, k: int = 10):
    # Phate Distance Matrix
    phate_op = phate.PHATE(random_state=1,
                           verbose=False,
                           n_components=2,
                           knn=k).fit(X)
    diff_pot = phate_op.diff_potential  # -log(P^t)

    assert diff_pot.shape[0] == X.shape[0]

    phate_distance = pairwise_distances(diff_pot)

    # Normalize
    D = (np.sum(phate_distance, axis=1))**0.5
    P = np.divide(phate_distance, D)
    P = (np.divide(P.T, D)).T
    # Deg = np.diag((1 / np.sum(phate_distance, axis=1))**0.5)
    # P = Deg @ phate_distance @ Deg

    return P


# def compute_diffusion_matrix(X: np.array, k: int = 10):
#     magic_op = magic.MAGIC(random_state=1, knn=k, verbose=False)
#     _ = magic_op.fit_transform(X)
#     P = magic_op.graph.diff_op.toarray()
#     return P


def compute_diffusion_matrix(X: np.array, sigma: float = 10.0):
    '''
    Adapted from
    https://github.com/professorwug/diffusion_curvature/blob/master/diffusion_curvature/core.py

    Given input X returns a diffusion matrix P, as an numpy ndarray.
    Using the "anisotropic" kernel
    Inputs:
        X: a numpy array of size n x d
        sigma: a float
            conceptually, the neighborhood size of Gaussian kernel.
    Returns:
        P: a numpy array of size n x n that is the diffusion matrix

    NOTE: With a recent update (using anisotropic kernel instead of adaptive kernel),
          the input argument `k` becomes a dummy argument.
          It's not removed yet for backward compatibility.
    '''

    # Construct the distance matrix.
    D = pairwise_distances(X)

    # Gaussian kernel
    W = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-D**2) / (2 * sigma**2))

    # Anisotropic density normalization.
    # if density_norm_pow > 0:
    #     Deg = np.diag(1 / np.sum(W, axis=1)**density_norm_pow)
    #     W = Deg @ W @ Deg

    # Gaussian kernel to diffusion matrix
    Deg = np.diag(1 / np.sum(W, axis=1)**0.5)
    P = Deg @ W @ Deg

    return P


def estimate_gaussian_kernel_sigma(X: np.array):
    # Construct the distance matrix.
    D = pairwise_distances(X)
    sigma = median_heuristic(D)
    return sigma


def median_heuristic(
        D: np.ndarray,  # the distance matrix
):
    # estimate kernel bandwidth from distance matrix using the median heuristic
    # Get upper triangle from distance matrix (ignoring duplicates)
    h = D[np.triu_indices_from(D)]
    h = h**2
    h = np.median(h)
    nu = np.sqrt(h / 2)
    return nu

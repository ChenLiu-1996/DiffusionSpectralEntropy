import numpy as np
from information_utils import approx_eigvals, exact_eigvals
from diffusion import compute_diffusion_matrix
import os


def diffusion_spectral_entropy(embedding_vectors: np.array,
                               gaussian_kernel_sigma: int = 10,
                               t: int = 1,
                               chebyshev_approx: bool = False,
                               eigval_save_path: str = None,
                               verbose: bool = True):
    '''
    DSE over a set of N vectors, each of D dimensions.

    H(G) = - sum_i [eig_i^t log eig_i^t]

    where each `eig_i` is an eigenvalue of `P`,
    where `P` is the diffusion matrix computed on the data graph of the [N, D] vectors.

    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        t: int
            Power of diffusion matrix (equivalent to power of diffusion eigenvalues)
            <-> Iteration of diffusion process
            Usually small, e.g., 1 or 2.
            Can be adjusted per dataset.
            Rule of thumb: after powering eigenvalues to `t`, there should be approximately
                           1 percent of eigenvalues that remain larger than 0.01

        gaussian_kernel_sigma: int
            The bandwidth of Gaussian kernel (for computation of the diffusion matrix)
            Can be adjusted per the dataset.
            Increase if the data points are very far away from each other.

        chebyshev_approx: bool
            Whether or not to use Chebyshev moments for faster approximation of eigenvalues.
            Currently we DO NOT RECOMMEND USING THIS. Eigenvalues may be changed quite a bit.

        eigval_save_path: str
            If provided,
                (1) If running for the first time, will save the computed eigenvalues in this location.
                (2) Otherwise, if the file already exists, skip eigenvalue computation and load from this file.

        verbose: bool
            Whether or not to print progress to console.
    '''

    if eigval_save_path is not None and os.path.exists(eigval_save_path):
        eigvals = np.load(eigval_save_path)['eigvals']
        eigvals = eigvals.astype(np.float64)  # mitigate rounding error.
        if verbose: print('Pre-computed eigenvalues loaded.')

    else:
        if verbose: print('Computing diffusion matrix.')
        # Note that `K` is a symmetric matrix with the same eigenvalues as the diffusion matrix `P`.
        K = compute_diffusion_matrix(embedding_vectors,
                                     sigma=gaussian_kernel_sigma)
        if verbose: print('Diffusion matrix computed.')

        if verbose: print('Computing eigenvalues.')
        if chebyshev_approx:
            if verbose: print('Using Chebyshev approximation.')
            eigvals = approx_eigvals(K)
        else:
            eigvals = exact_eigvals(K)
        if verbose: print('Eigenvalues computed.')

        if eigval_save_path is not None:
            # Save eigenvalues.
            eigvals = eigvals.astype(np.float16)  # save disk space.
            with open(eigval_save_path, 'wb+') as f:
                np.savez(f, eigvals=eigvals)
            if verbose: print('Eigenvalues saved to %s' % eigval_save_path)

    # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
    eigvals = np.array(sorted(eigvals)[::-1])
    eigvals = eigvals[1:]

    # Eigenvalues may be negative. Only care about the magnitude, not the sign.
    eigvals = np.abs(eigvals)

    # Power eigenvalues to `t` to mitigate effect of noise.
    eigvals = eigvals**t

    prob = eigvals / eigvals.sum()
    prob = prob + np.finfo(float).eps

    DSE = -np.sum(prob * np.log2(prob))

    return DSE


def classic_shannon_entropy(embedding_vectors: np.array,
                            num_bins_per_dim: int = 2):
    '''
    Classic Shannon Entropy over a set of N vectors, each of D dimensions.

    H(X) = - sum_i [p(x) log p(x)]

    where each p(x) is the probability density of a histogram bin, after some sort of binning.
    '''
    vecs = embedding_vectors.copy()

    # Min-Max scale each dimension.
    vecs = (vecs - np.min(vecs, axis=0)) / (np.max(vecs, axis=0) -
                                            np.min(vecs, axis=0))

    # Bin along each dimension.
    bins = np.linspace(0, 1, num_bins_per_dim + 1)[:-1]
    vecs = np.digitize(vecs, bins=bins)

    # Count probability.
    counts = np.unique(vecs, axis=0, return_counts=True)[1]
    prob = counts / np.sum(counts)
    prob = prob + np.finfo(float).eps
    CSE = -np.sum(prob * np.log2(prob))

    return CSE

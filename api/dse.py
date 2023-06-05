import numpy as np
from information_utils import approx_eigvals, exact_eigvals
from diffusion import compute_diffusion_matrix
import os


def diffusion_spectral_entropy(embedding_vectors: np.array,
                               gaussian_kernel_sigma: int = 10,
                               t: int = 1,
                               chebyshev_approx: bool = False,
                               eigval_save_path: str = None,
                               eigval_save_precision: np.dtype = np.float16,
                               verbose: bool = True):
    '''
    DSE over a set of N vectors, each of D dimensions.

    DSE = - sum_i [eig_i^t log eig_i^t]

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

        eigval_save_precision: np.dtype
            We use `np.float16` by default to reduce storage space required.
            For best precision, use `np.float64` instead.

        verbose: bool
            Whether or not to print progress to console.
    '''

    if eigval_save_path is not None and os.path.exists(eigval_save_path):
        if verbose:
            print('Loading pre-computed eigenvalues from %s' %
                  eigval_save_path)
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
            os.makedirs(os.path.dirname(eigval_save_path), exist_ok=True)
            # Save eigenvalues.
            eigvals = eigvals.astype(
                eigval_save_precision)  # reduce storage space.
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

    CSE = - sum_i [p(x) log p(x)]

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


if __name__ == '__main__':
    print('Testing Diffusion Spectral Entropy.')
    print('\n1st run, random vecs, without saving eigvals.')
    embedding_vectors = np.random.uniform(0, 1, (1000, 256))
    DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors)
    print('DSE =', DSE)

    print('\n2nd run, random vecs, saving eigvals (np.float16).')
    tmp_path = './test_dse_eigval.npz'
    embedding_vectors = np.random.uniform(0, 2, (1000, 256))
    DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                     eigval_save_path=tmp_path)
    print('DSE =', DSE)

    print(
        '\n3rd run, loading eigvals from 2nd run. May be slightly off due to float16 saving.'
    )
    embedding_vectors = None  # does not matter, will be ignored anyways
    DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                     eigval_save_path=tmp_path)
    print('DSE =', DSE)
    os.remove(tmp_path)

    print('\n4th run, random vecs, saving eigvals (np.float64).')
    embedding_vectors = np.random.uniform(0, 3, (1000, 256))
    DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                     eigval_save_path=tmp_path,
                                     eigval_save_precision=np.float64)
    print('DSE =', DSE)

    print('\n5th run, loading eigvals from 4th run. Shall be identical.')
    embedding_vectors = None  # does not matter, will be ignored anyways
    DSE = diffusion_spectral_entropy(embedding_vectors=embedding_vectors,
                                     eigval_save_path=tmp_path)
    print('DSE =', DSE)
    os.remove(tmp_path)

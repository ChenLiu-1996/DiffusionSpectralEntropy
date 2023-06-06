import numpy as np
from dse import diffusion_spectral_entropy
from sklearn.cluster import SpectralClustering
import random


def diffusion_spectral_mutual_information(
        embedding_vectors: np.array,
        reference_vectors: np.array,
        reference_discrete: bool = None,
        gaussian_kernel_sigma: int = 10,
        t: int = 1,
        chebyshev_approx: bool = False,
        num_repetitions: int = 5,
        n_clusters: int = 10,
        precomputed_clusters: np.array = None,
        classic_shannon_entropy: bool = False,
        num_bins_per_dim: int = 2,
        random_seed: int = 0,
        verbose: bool = False):
    '''
    DSMI between two sets of random variables.
    The first (`embedding_vectors`) must be a set of N vectors each of D dimension.
    The second (`reference_vectors`) must be a set of N vectors each of D' dimension.
        D is not necessarily the same as D'.
        In some common cases, we may have the following as `reference_vectors`
            - class labels (D' == 1) of shape [N, 1]
            - flattened input signals/images of shape [N, D']

    DSE(A; B) = DSE(A) - DSE(A | B)
        where DSE is the diffusion spectral entropy.

    DSE(A | B) = sum_i [p(B = b_i) DSE(A | B = b_i)]
        where i = 0,1,...,m
            m = number of categories in random variable B
            if B itself is a discrete variable (e.g., class label), this is straightforward
            otherwise, we can use spectral clustering to create discrete categories/clusters in B

    For numerical consistency, instead of computing DSE(A) on all data points of A,
    we estimate it from a subset of A, with the size of subset equal to {B = b_i}.

    The final computation is:

    DSE(A; B) = DSE(A) - DSE(A | B) = sum_i [p(B = b_i) (DSE(A*) - DSE(A | B = b_i))]
        where A* is a subsampled version of A, with len(A*) == len(B = b_i).

    args:
        embedding_vectors: np.array of shape [N, D]
            N: number of data points / samples
            D: number of feature dimensions of the neural representation

        reference_vectors: np.array of shape [N, D']
            N: number of data points / samples
            D': number of feature dimensions of the neural representation or input/output variable

        reference_discrete: bool
            Whether `reference_vectors` is discrete or continuous.
            This determines whether or not we perform clustering/binning on `reference_vectors`.
            NOTE: If True, we assume D' == 1. Common case: `reference_vectors` is the discrete class labels.
            If not provided, will be inferred from `reference_vectors`.

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

        num_repetitions: int
            Number of repetition during DSE(A*) estimation.
            The variance is usually low, so a small number shall suffice.

        random_seed: int
            Random seed. For DSE(A*) estimation repeatability.

        n_clusters: int
            Number of clusters for `reference_vectors`.
            Only used when `reference_discrete` is False (`reference_vectors` is not discrete).
            If D' == 1 --> will use scalar binning.
            If D' > 1  --> will use spectral clustering.

        precomputed_clusters: np.array
            If provided, will directly use it as the cluster assignments for `reference_vectors`.
            Only used when `reference_discrete` is False (`reference_vectors` is not discrete).
            NOTE: When you have a fixed set of `reference_vectors` (e.g., a set of images),
            you probably want to only compute the spectral clustering once, and recycle the computed
            clusters for subsequent DSMI computations.

        classic_shannon_entropy: bool
            Whether or not we use CSE to replace DSE in the computation.
            NOTE: If true, the resulting mutual information will be CSMI instead of DSMI.

        num_bins_per_dim: int
            Number of bins per feature dim.
            Only relevant to CSE (i.e., `classic_shannon_entropy` is True).

        verbose: bool
            Whether or not to print progress to console.
    '''

    # Reshape from [N, ] to [N, 1].
    if len(reference_vectors.shape) == 1:
        reference_vectors = reference_vectors.reshape(
            reference_vectors.shape[0], 1)

    N_embedding, _ = embedding_vectors.shape
    N_reference, D_reference = reference_vectors.shape

    if N_embedding != N_reference:
        if verbose:
            print(
                'WARNING: DSMI embedding and reference do not have the same N: %s vs %s'
                % (N_embedding, N_reference))

    if reference_discrete is None:
        # Infer whether `reference_vectors` is discrete.
        # Criteria: D' == 1 and `reference_vectors` is an integer type.
        reference_discrete = D_reference == 1 \
            and np.issubdtype(
            reference_vectors.dtype, np.integer)

    #
    '''STEP 1. Prepare the category/cluster assignments.'''

    if reference_discrete:
        # `reference_vectors` is expected to be discrete class labels.
        assert D_reference == 1, \
            'DSMI `reference_discrete` is set to True, but shape of `reference_vectors` is not [N, 1].'
        precomputed_clusters = reference_vectors

    elif D_reference == 1:
        # `reference_vectors` is a set of continuous scalars.
        # Perform scalar binning if cluster assignments are not provided.
        if precomputed_clusters is None:
            vecs = reference_vectors.copy()
            # Min-Max scale each dimension.
            vecs = (vecs - np.min(vecs, axis=0)) / (np.max(vecs, axis=0) -
                                                    np.min(vecs, axis=0))
            # Bin along each dimension.
            bins = np.linspace(0, 1, n_clusters + 1)[:-1]
            vecs = np.digitize(vecs, bins=bins)
            precomputed_clusters = vecs

    else:
        # `reference_vectors` is a set of continuous vectors.
        # Perform spectral clustering if cluster assignments are not provided.
        if precomputed_clusters is None:
            cluster_op = SpectralClustering(
                n_clusters=n_clusters,
                affinity='nearest_neighbors',
                assign_labels='cluster_qr',
                random_state=0).fit(reference_vectors)
            precomputed_clusters = cluster_op.labels_

    clusters_list, cluster_cnts = np.unique(precomputed_clusters,
                                            return_counts=True)

    #
    '''STEP 2. Compute DSMI.'''
    MI_by_class = []

    for cluster_idx in clusters_list:
        # DSE(A | B = b_i)
        inds = (precomputed_clusters == cluster_idx).reshape(-1)
        embeddings_curr_class = embedding_vectors[inds, :]

        entropy_AgivenB_curr_class = diffusion_spectral_entropy(
            embedding_vectors=embeddings_curr_class,
            gaussian_kernel_sigma=gaussian_kernel_sigma,
            t=t,
            chebyshev_approx=chebyshev_approx,
            classic_shannon_entropy=classic_shannon_entropy,
            num_bins_per_dim=num_bins_per_dim)

        # DSE(A*)
        if random_seed is not None:
            random.seed(random_seed)
        entropy_A_estimation_list = []
        for _ in np.arange(num_repetitions):
            rand_inds = np.array(
                random.sample(range(precomputed_clusters.shape[0]),
                              k=np.sum(precomputed_clusters == cluster_idx)))
            embeddings_random_subset = embedding_vectors[rand_inds, :]

            entropy_A_subsample_rep = diffusion_spectral_entropy(
                embedding_vectors=embeddings_random_subset,
                gaussian_kernel_sigma=gaussian_kernel_sigma,
                t=t,
                chebyshev_approx=chebyshev_approx,
                classic_shannon_entropy=classic_shannon_entropy,
                num_bins_per_dim=num_bins_per_dim)
            entropy_A_estimation_list.append(entropy_A_subsample_rep)

        entropy_A_estimation = np.mean(entropy_A_estimation_list)

        MI_by_class.append((entropy_A_estimation - entropy_AgivenB_curr_class))

    mutual_information = np.sum(cluster_cnts / np.sum(cluster_cnts) *
                                np.array(MI_by_class))

    return mutual_information, precomputed_clusters


if __name__ == '__main__':
    print('Testing Diffusion Spectral Mutual Information.')
    print('\n1st run. DSMI, Embeddings vs discrete class labels.')
    embedding_vectors = np.random.uniform(0, 1, (1000, 256))
    class_labels = np.uint8(np.random.uniform(0, 11, (1000, 1)))
    DSMI, _ = diffusion_spectral_mutual_information(
        embedding_vectors=embedding_vectors, reference_vectors=class_labels)
    print('DSMI =', DSMI)

    print('\n2nd run. DSMI, Embeddings vs continuous scalars')
    embedding_vectors = np.random.uniform(0, 1, (1000, 256))
    continuous_scalars = np.random.uniform(-1, 1, (1000, 1))
    DSMI, _ = diffusion_spectral_mutual_information(
        embedding_vectors=embedding_vectors,
        reference_vectors=continuous_scalars)
    print('DSMI =', DSMI)

    print('\n3rd run. DSMI, Embeddings vs Input Image')
    embedding_vectors = np.random.uniform(0, 1, (1000, 256))
    input_image = np.random.uniform(-1, 1, (1000, 3, 32, 32))
    input_image = input_image.reshape(input_image.shape[0], -1)
    DSMI, _ = diffusion_spectral_mutual_information(
        embedding_vectors=embedding_vectors, reference_vectors=input_image)
    print('DSMI =', DSMI)

    print('\n4th run. DSMI, Classification dataset.')
    from sklearn.datasets import make_classification
    embedding_vectors, class_labels = make_classification(n_samples=1000,
                                                          n_features=5)
    DSMI, _ = diffusion_spectral_mutual_information(
        embedding_vectors=embedding_vectors, reference_vectors=class_labels)
    print('DSMI =', DSMI)

    print('\n5th run. CSMI, Classification dataset.')
    embedding_vectors, class_labels = make_classification(n_samples=1000,
                                                          n_features=5)
    CSMI, _ = diffusion_spectral_mutual_information(
        embedding_vectors=embedding_vectors,
        reference_vectors=class_labels,
        classic_shannon_entropy=True)
    print('CSMI =', CSMI)

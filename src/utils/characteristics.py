from typing import List

import numpy as np
from tqdm import tqdm

from diffusion import compute_diffusion_matrix
from DiffusionEMD.diffusion_emd import estimate_dos


def simple_bin(cond_x: np.array, num_digit: int):
    '''
        put N of D-dim vectors into discrete bins
        the final number of unique D-dim vectors is M
    Args:
        cond_x: [N, D]

    Returns:
        assignment: [N, 1]
        cnts: List of length M, each indicating the number of rows in that specific unique row

    '''
    # minmax normalize to [0,1]
    cond_x = (cond_x - np.min(cond_x)) / (np.max(cond_x) - np.min(cond_x))
    bins = np.linspace(0, 1, num_digit, dtype='float32')
    # bin each element, [N x d_2]
    digitized_cond_x = np.digitize(cond_x, bins=bins, right=False)

    # turn d_2 feature vector to 1-dim [N x d_2] -> [N x 1], for the purpose of using np.unique
    cond_rows = np.ascontiguousarray(digitized_cond_x).view(
        np.dtype(
            (np.void,
             digitized_cond_x.dtype.itemsize * digitized_cond_x.shape[1])))
    _, assignments, cnts = np.unique(cond_rows,
                                     return_index=False,
                                     return_inverse=True,
                                     return_counts=True)

    return assignments, cnts


def comp_diffusion_embedding(X: np.array, knn: int):
    '''
        Compute diffusion embedding of X
    Args:
        X: [N, D]

    Returns:
        diff_embed: [N, N]
    '''
    # Diffusion matrix
    diffusion_matrix = compute_diffusion_matrix(X, k=knn)
    eigenvalues_P, eigenvectors_P = np.linalg.eig(diffusion_matrix)

    # Sort eigenvalues
    sorted_idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[sorted_idx]
    eigenvectors_P = eigenvectors_P[:, sorted_idx]
    # Diffusion map embedding
    diff_embed = eigenvectors_P @ np.diag((eigenvalues_P**0.5))

    return diff_embed


def mutual_information(orig_x: np.array,
                       cond_x: np.array,
                       knn: int,
                       class_method: str = 'bin',
                       num_digit: int = 2,
                       num_spectral: int = None,
                       diff_embed: np.array = None,
                       num_clusters: int = 100,
                       orig_entropy: float = None):
    '''
        To compute the conditioned entropy H(orig_x|cond_x), we categorize the cond_x into discrete classes,
        and then compute the VNE for subgraphs of orig_x based on classes.

        class_method: 'bin', 'spectral_bin', 'kmeans'

        'bin': Bin directly in vector space, adapted from https://github.com/artemyk/ibsgd/blob/master/simplebinmi.py
        'spectral bin': Bin in spectral space: first convert cond_x to Diffusion Map coords, then bin

        orig_x: [N x d_1]
        cond_x: [N x d_2]
    '''
    assert orig_x.shape[0] == cond_x.shape[0]

    # Categorize the cond_x into discrete classes
    cond_classes = None

    if class_method == 'bin':
        '''
            Bin in vector space:
        '''

        cond_classes, classes_cnts = simple_bin(cond_x, num_digit=num_digit)

    elif class_method == 'spectral_bin':
        '''
            Bin in spectral space
        '''
        # diffusion map coords of cond_x
        if num_spectral is None:
            num_spectral = min(cond_x.shape[1], cond_x.shape[0])
        if diff_embed is None:
            diff_embed = comp_diffusion_embedding(X=cond_x)

        # Top components
        diff_embed = diff_embed[:, :num_spectral]

        # simple bin on the diffusion map coords
        cond_classes, classes_cnts = simple_bin(cond_x=diff_embed,
                                                num_digit=num_digit)

    elif class_method == 'kmeans':
        '''
            Kmeans
        '''
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_clusters, random_state=0,
                        n_init="auto").fit(cond_x)
        cond_classes = kmeans.labels_
        _, classes_cnts = np.unique(cond_classes, return_counts=True)
    elif class_method == 'kspectral':
        '''
            SpectralClustering,
            clustering to a projection of the normalized Laplacian.
        '''
        from sklearn.cluster import SpectralClustering
        return NotImplementedError

    classes_list = np.unique(cond_classes, return_counts=False)
    assert cond_classes.shape[0] == orig_x.shape[0]

    # Compute VNE of subgraphs of orig_x according to cond_classes
    vne_by_classes = []
    for class_idx in tqdm(classes_list):
        inds = (cond_classes == class_idx).reshape(-1)
        samples = orig_x[inds, :]

        # if samples.shape[0] <= knn, dynamic adjust knn to samples.shape[0]
        min_node_num = 2
        if samples.shape[0] < min_node_num:
            s_vne = 0.0
        else:
            # Diffusion Matrix
            s_diffusion_matrix = compute_diffusion_matrix(samples, k=knn)
            # Eigenvalues
            s_eigenvalues_P = np.linalg.eigvals(s_diffusion_matrix)
            # Von Neumann Entropy
            s_vne = von_neumann_entropy(s_eigenvalues_P)

        vne_by_classes.append(s_vne)

    # H(orig_x|cond_x)
    conditioned_entropy = np.sum(
        np.array(classes_cnts) / np.sum(classes_cnts) *
        np.array(vne_by_classes))

    if orig_entropy is None:
        # Diffusion Matrix
        diffusion_matrix = compute_diffusion_matrix(samples, k=knn)
        # Eigenvalues
        eigenvalues_P = np.linalg.eigvals(diffusion_matrix)
        # Von Neumann Entropy
        orig_entropy = von_neumann_entropy(eigenvalues_P)

    mi = orig_entropy - conditioned_entropy

    return mi, conditioned_entropy, len(classes_list)


def mutual_information_per_class_random_sample(embeddings: np.array,
                                               labels: np.array,
                                               num_repetitions: int = 5,
                                               knn: int = 10):
    '''
    Randomly assign class labels to entire embeds graph
    for computing unconditioned entropy

    Using the formula:
        I(Z; Y) = H(Z) - H(Z | Y)
            H(Z | Y) is directly computed
            H(Z) is estimated by sampling #pts = count(Y=y) from Z
        Note: the key is that we don't use the entire graph
              (i.e., all Z) to compute H(Z).

    Args:
        embeddings: [N,D]
        labels: [N,1]
    Returns:
        mi: scaler
    '''
    classes_list, class_cnts = np.unique(labels, return_counts=True)
    mi_by_class = []

    for class_idx in tqdm(classes_list):
        inds = (labels == class_idx).reshape(-1)
        Z_curr_class = embeddings[inds, :]

        #
        ''' Class conditioned entropy H(Z | Y)'''
        # Diffusion Matrix
        diffusion_matrix_curr_class = compute_diffusion_matrix(Z_curr_class,
                                                               k=knn)
        # Eigenvalues
        eigenvalues_curr_class = np.linalg.eigvals(diffusion_matrix_curr_class)
        # Von Neumann Entropy
        H_ZgivenY = von_neumann_entropy(eigenvalues_curr_class)

        #
        ''' Unconditioned entropy H(Z), estimated by randomly sampling the same number of points'''
        H_Z_list = []
        for _ in np.arange(num_repetitions):
            rand_inds = np.random.choice(labels.shape[0],
                                         size=Z_curr_class.shape[0],
                                         replace=False)
            Z_random = embeddings[rand_inds, :]
            # Diffusion Matrix
            diffusion_matrix_random_set = compute_diffusion_matrix(Z_random,
                                                                   k=knn)
            # Eigenvalues
            eigenvalues_random_set = np.linalg.eigvals(
                diffusion_matrix_random_set)
            # Von Neumann Entropy
            H_Z_rep = von_neumann_entropy(eigenvalues_random_set)
            H_Z_list.append(H_Z_rep)

        H_Z = np.mean(H_Z_list)

        mi_by_class.append((H_Z - H_ZgivenY))

    mi = np.sum(class_cnts / np.sum(class_cnts) * np.array(mi_by_class))

    return mi


def mutual_information_per_class(eigs: np.array,
                                 vne_by_class: List[np.float64],
                                 n_by_class: List[int],
                                 unconditioned_entropy: float = None):
    '''
    Using the formula:
    I(Z; Y) = H(Z) - H(Z | Y)
        H(Z | Y) is directly computed
        H(Z) is directly computed over the entire graph (i.e., all Z).
    '''

    # H(h_m)
    if unconditioned_entropy is None:
        unconditioned_entropy = von_neumann_entropy(eigs)

    # H(h_m|Y), Y is the class
    conditioned_entropy = np.sum(
        np.array(n_by_class) / np.sum(n_by_class) * np.array(vne_by_class))

    # I(h_m; Y)
    mi = unconditioned_entropy - conditioned_entropy

    return mi


# def von_neumann_entropy(eigs: np.array, eps: float = 1e-3):
#     eigenvalues = eigs.copy()

#     eigenvalues = np.array(sorted(eigenvalues)[::-1])

#     # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
#     eigenvalues = eigenvalues[1:]

#     # Drop the close-to-zero eigenvalue(s).
#     # This also takes care of the numerical rounding issues where
#     # some eigenvalues gets slightly negative.
#     eigenvalues = eigenvalues[eigenvalues >= eps]

#     prob = eigenvalues / eigenvalues.sum()
#     prob = prob + np.finfo(float).eps

#     return -np.sum(prob * np.log(prob))

# def von_neumann_entropy(eigs: np.array, eps: float = 1e-3):
#     eigenvalues = eigs.copy()

#     eigenvalues = np.array(sorted(eigenvalues)[::-1])

#     # Shift the negative eigenvalue(s) that occurred due to rounding errors.
#     if eigenvalues.min() < 0:
#         eigenvalues -= eigenvalues.min()

#     # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
#     eigenvalues = eigenvalues[1:]

#     # Drop the close-to-zero eigenvalue(s).
#     eigenvalues = eigenvalues[eigenvalues >= eps]

#     prob = eigenvalues / eigenvalues.sum()
#     prob = prob + np.finfo(float).eps

#     return -np.sum(prob * np.log(prob))

# def von_neumann_entropy(eigs: np.array, eps: float = 1e-3):
# def von_neumann_entropy(eigs: np.array, num_bins: int = 1000):
# THIS IS BAD!
#     eigenvalues = eigs.copy()
#     eigenvalues = eigenvalues.astype(np.float64)  # mitigates rounding error.

#     eigenvalues = np.array(sorted(eigenvalues)[::-1])

#     # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
#     eigenvalues = eigenvalues[1:]

#     # Eigenvalues may be negative. Only care about the magnitude, not the sign.
#     eigenvalues = np.abs(eigenvalues)

#     bins = np.linspace(-1, 1, num_bins)
#     frequency, _ = np.histogram(eigenvalues, bins=bins)
#     density = frequency / np.sum(frequency)

#     prob = density + np.finfo(float).eps

#     return -np.sum(prob * np.log2(prob))


def approx_eigvals(A: np.array, filter_thr: float = 1e-3):
    matrix = A.copy()
    N = matrix.shape[0]

    if filter_thr is not None:
        matrix[np.abs(matrix) < filter_thr] = 0

    eigs, cdf = estimate_dos(matrix)
    pdf = np.zeros_like(cdf)
    for i in range(len(cdf) - 1):
        pdf[i] = cdf[i + 1] - cdf[i]

    counts = N * pdf / np.sum(pdf)

    eigenvalues = []
    for i, count in enumerate(counts):
        if np.round(count) > 0:
            for _ in range(int(np.round(count))):
                eigenvalues.append(eigs[i])

    eigenvalues = np.array(eigenvalues)

    return eigenvalues


def von_neumann_entropy(eigs: np.array, noise_eigval_thr: float = 1e-3):
    eigenvalues = eigs.copy()
    eigenvalues = eigenvalues.astype(np.float64)  # mitigates rounding error.

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
    eigenvalues = eigenvalues[1:]

    # Eigenvalues may be negative. Only care about the magnitude, not the sign.
    eigenvalues = np.abs(eigenvalues)

    # Drop the trivial eigenvalues that are corresponding to noise eigenvectors.
    eigenvalues[eigenvalues < noise_eigval_thr] = 0

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log2(prob))

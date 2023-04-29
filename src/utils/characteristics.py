from typing import List

import numpy as np
from tqdm import tqdm

from diffusion import compute_diffusion_matrix

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
    cond_rows = np.ascontiguousarray(digitized_cond_x).view(np.dtype(
        (np.void, digitized_cond_x.dtype.itemsize * digitized_cond_x.shape[1])))
    _, assignments, cnts = np.unique(
        cond_rows, return_index=False, return_inverse=True, return_counts=True)
    
    return assignments, cnts

def diffusion_embedding(X: np.array, num_components: int, knn: int):
    '''
        Compute diffusion embedding of X, taking first num_components eigenvectors
    Args:
        X: [N, D]

    Returns:
        diff_embed: [N, num_components]
    '''
     # Diffusion matrix
    diffusion_matrix = compute_diffusion_matrix(X, k=knn)
    eigenvalues_P, eigenvectors_P = np.linalg.eig(diffusion_matrix)

    # Sort eigenvalues & take top num_components
    sorted_idx = np.argsort(eigenvalues_P)[::-1]
    eigenvalues_P = eigenvalues_P[sorted_idx]
    eigenvectors_P = eigenvectors_P[:, sorted_idx]

    W = eigenvalues_P[:num_components]
    V = eigenvectors_P[:, num_components]
    
    
    # Diffusion map embedding
    diff_embed = V @ np.diag((W**0.5))
    assert diff_embed.shap[1] == num_components

    return diff_embed



def mutual_information(orig_x: np.array,
                       cond_x: np.array,
                       knn: int,
                       class_method: str = 'bin',
                       num_digit: int = 2,
                       num_spectral: int = None,
                       orig_entropy: float = None):
    '''
        To compute the conditioned entropy H(orig_x|cond_x), we categorize the cond_x into discrete classes,
        and then compute the VNE for subgraphs of orig_x based on classes.

        class_method: 'bin', 'spectral_bin', 'precompute', 'kmeans'

        'bin': Bin directly in vector space, adapted from https://github.com/artemyk/ibsgd/blob/master/simplebinmi.py
        'spectral bin': Bin in spectral space: first convert cond_x to Diffusion Map coords, then bin

        orig_x: [N x d_1]
        cond_x: [N x d_2]
    '''
    assert orig_x.shape[0] == cond_x.shape[0]

    # Categorize the cond_x into discrete classes
    cond_classes = None

    if class_method == 'precompute':
        cond_classes = cond_x
    elif class_method == 'bin':
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
        diff_embed = diffusion_embedding(X=cond_x, num_components=num_spectral)

        # simple bin on the diffusion map coords
        cond_classes, classes_cnts = simple_bin(cond_x=diff_embed, num_digit=num_digit)
        
    elif class_method == 'kmeans':
        return NotImplementedError


    classes_list = np.unique(cond_classes, return_counts=False)
    print('classes_list len :', len(classes_list), ' cond_x.shape[1]: ', cond_x.shape[1])
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


def mutual_information_per_class(eigs: np.array,
                       vne_by_class: List[np.float64],
                       n_by_class: List[int],
                       unconditioned_entropy: float = None):
    # H(h_m)
    if unconditioned_entropy is None:
        unconditioned_entropy = von_neumann_entropy(eigs)

    # H(h_m|Y), Y is the class
    conditioned_entropy = np.sum(
        np.array(n_by_class) / np.sum(n_by_class) *
        np.array(vne_by_class))

    # I(h_m; Y)
    mi = unconditioned_entropy - conditioned_entropy

    return mi


def von_neumann_entropy(eigs: np.array, eps: float = 1e-3):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Shift the negative eigenvalue(s) that occurred due to rounding errors.
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()

    # Drop the trivial eigenvalue corresponding to the indicator eigenvector.
    eigenvalues = eigenvalues[eigenvalues <= 1 - eps]

    # Drop the close-to-zero eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues >= eps]

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))

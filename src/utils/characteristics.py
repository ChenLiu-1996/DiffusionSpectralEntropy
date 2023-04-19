from typing import List

import numpy as np


def mutual_information(eigs: np.array,
                       eigs_by_class: List[np.array],
                       n_by_class: List[int],
                       unconditioned_entropy: float = None):
    # H(h_m)
    if unconditioned_entropy is None:
        unconditioned_entropy = von_neumann_entropy(eigs)

    # H(h_m|Y), Y is the class
    conditioned_entropy_list = [
        von_neumann_entropy(eig) for eig in eigs_by_class
    ]
    conditioned_entropy = np.sum(
        np.array(n_by_class) / np.sum(n_by_class) *
        np.array(conditioned_entropy_list))

    # I(h_m; Y)
    mi = unconditioned_entropy - conditioned_entropy

    return mi


def von_neumann_entropy(eigs: np.array, eps: float = 1e-2):
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

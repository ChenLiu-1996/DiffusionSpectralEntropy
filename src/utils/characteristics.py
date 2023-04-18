from typing import List
import numpy as np


def mutual_information(eigs: np.array,
                       eigs_by_class: List[np.array],
                       n_by_class: List[int],
                       eps: float = 1e-3):
    return NotImplementedError


def von_neumann_entropy(eigs: np.array, eps: float = 1e-3):
    eigenvalues = eigs.copy()

    eigenvalues = np.array(sorted(eigenvalues)[::-1])

    # Drop the trivial eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues <= 1 - eps]

    # Shift the negative eigenvalue(s) that occurred due to rounding errors.
    if eigenvalues.min() < 0:
        eigenvalues -= eigenvalues.min()

    # Remove the close-to-zero eigenvalue(s).
    eigenvalues = eigenvalues[eigenvalues >= eps]

    prob = eigenvalues / eigenvalues.sum()
    prob = prob + np.finfo(float).eps

    return -np.sum(prob * np.log(prob))

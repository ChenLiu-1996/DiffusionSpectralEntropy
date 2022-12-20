# A standalone python script to calculate the diffusion curvature of a supplied point cloud.
# Prints curvature values, and also saves them to a file with the same name as the input, but with '-curvatures' appended 
import numpy as np
from sklearn.metrics import pairwise_distances
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse

parser = argparse.ArgumentParser(description='Diffusion Curvature Arguments') # collect arguments passed to file
parser.add_argument('--kernel', type=str,
                    help='Kernel type for diffusion matrix. Adaptive/anisotropic/adaptive anisotropic',default='anisotropic')
parser.add_argument('--file',type=str,help="file path to point cloud (assumes 1 pt per line)")
parser.add_argument('--t',type=int,help="Number of steps of diffusion to take",default=8)
parser.add_argument('--aperture',type=int,help="Size of local neighborhood in which to sum diffusion probabilities",default=20)
parser.add_argument('--sigma',type=float,help="Kernel bandwidth for anisotropic kernel",default=0.7)
parser.add_argument('--k', type=int, help='number of neighbors to be used by adaptive kernel',default=20)
args = parser.parse_args()

def DiffusionMatrix(X, kernel_type = "fixed", sigma = 0.7, k = 20, alpha = 0.5, affinity_matrix_only=False):
    """
    Given input X returns a diffusion matrix P, as an numpy ndarray.
    X is a numpy array of size n x d
    kernel_type is a string, either "fixed" or "adaptive" or "anisotropic" or "adaptive anisotropic"
    sigma is the non-adaptive gaussian kernel parameter
    k is the adaptive kernel parameter
    Returns:
    P is a numpy array of size n x n that is the diffusion matrix
    """
    # construct the distance matrix
    D = pairwise_distances(X)
    # make the affinity matrix
    if kernel_type == "fixed":
        W = (1/sigma*np.sqrt(2*np.pi))*np.exp(-D**2/(2*sigma**2))
    elif kernel_type == "adaptive" or kernel_type == "adaptive anisotropic":
        # Get the distance to the kth neighbor
        distance_to_k_neighbor = np.partition(D,k)[:,k]
        # Populate matrices with this distance for easy division.
        div1 = np.ones(len(D))[:,None] @ distance_to_k_neighbor[None,:]
        div2 = distance_to_k_neighbor[:,None] @ np.ones(len(D))[None,:]
        # print("Distance to kth neighbors",distance_to_k_neighbor)
        # compute the gaussian kernel with an adaptive bandwidth
        W = (1/2*np.sqrt(2*np.pi))*(np.exp(-D**2/(2*div1**2))/div1 + np.exp(-D**2/(2*div2**2))/div2)
        if kernel_type == "adaptive anisotropic":
            # Additional normalization step for density
            D = np.diag(1/np.sum(W,axis=1))
            W = D @ W @ D
    elif kernel_type == "nearest neighbor":
        pass
    elif kernel_type == "anisotropic":
        W1 = np.exp(-D**2/(2*sigma**2))
        D = np.diag(1/np.sum(W1,axis=1))
        W = D @ W1 @ D
    elif kernel_type == "alpha-decay":
        distance_to_k_neighbor = tf.nn.top_k(D, k = k, sorted = True).values[:,-1]
        distance_to_k_neighbor = tf.cast(distance_to_k_neighbor,tf.float32)
        D = tf.cast(D, tf.float32)
        div1 = tf.linalg.matmul(tf.ones(len(D))[:,None], distance_to_k_neighbor[None,:])
        div2 = tf.linalg.matmul(distance_to_k_neighbor[:,None],tf.ones(len(D))[None,:])
        W = 0.5*(tf.exp(-(D/div1)**alpha) + tf.exp(-(D/div2)**alpha))
    else:
        raise ValueError("kernel_type must be either 'fixed' or 'adaptive'")
    if affinity_matrix_only:
        return W
    # turn affinity matrix into diffusion matrix
    D = np.diag(1/np.sum(W,axis=1))
    P = D @ W
    return P

def curvature(P, diffusion_powers=8, aperture = 20, smoothing=1, verbose = False, return_density = False, dynamically_adjusting_neighborhood = False, precomputed_powered_P = None, non_lazy_diffusion=False, restrict_diffusion_to_k_neighborhood=None, avg_transition_probability=True, use_min_threshold = False):
    """Diffusion Laziness Curvature
    Estimates curvature by measuring the amount of mass remaining within an initial neighborhood after t steps of diffusion. Akin to measuring the laziness of a random walk after t steps.

    Parameters
    ----------
    P : n x n ndarray
        The diffusion matrix of the graph
    diffusion_powers : int, optional
        Number of steps of diffusion to take before measuring the laziness, by default 8
    aperture : int, optional
        The size of the initial neighborhood, from which the percentage of mass remaining in this neighborhood is calculated, by default 20
    smoothing : int, optional
        Amount of smoothing to apply. Currently works by multiplying the raw laziness values with the diffusion operator, as a kind of iterated weighted averaging; by default 1
    verbose : bool, optional
        Print diagnostics, by default False
    return_density : bool, optional
        Return the number of neighbors each point shares, by default False
    dynamically_adjusting_neighborhood : bool, optional
        Whether to give each point the same initial neighborhood size, by default False
    precomputed_powered_P : ndarray, optional
        Optionally pass a precomputed powered diffusion operator, to speed up computation, by default None
    avg_transition_probability: bool, default True
        Use the definition of diffusion curvature in which the summed transition probabilities are divided by the total number of points in the aperture neighborhood.
        As a result, gives not the summed "return probability within the neighborhood" but the average return probability to each point in the aperture neighborhood.
        This formulation of diffusion curvature was used in proof given in our NeurIPS 2022 paper.

    Returns
    -------
    length n array
        The laziness curvature values for each point
    """
    # the aperture sets the size of the one-hop neighborhood
    # the aperture parameter is the average number of neighbors to include, based off of the sorted diffusion values
    # Set thresholds as the kth largest diffusion value, presumed to be held by the kth nearest neighbor.
    thresholds = np.partition(P,-aperture)[:,-aperture]
    # thresholds = np.sort(P)[:,-aperture]
    if verbose: print(thresholds)
    if dynamically_adjusting_neighborhood:
        P_thresholded = (P >= thresholds[:,None]).astype(int)
    else:
        if use_min_threshold:
            P_threshold = np.min(thresholds)
        else:
            P_threshold = np.mean(thresholds) # TODO could also use min
        P_thresholded = (P >= P_threshold).astype(int)
        if verbose: print("Derived threshold ",P_threshold)

    if verbose: print(np.sum(P_thresholded,axis=1))
    if verbose: print("Performing matrix powers...")

    if precomputed_powered_P is not None:
        P_powered = precomputed_powered_P
    elif non_lazy_diffusion:
        print("Removing self-diffusion")
        P_zero_diagonal = (np.ones_like(P) - np.diag(np.ones(len(P))))*P
        D = np.diag(1/np.sum(P_zero_diagonal,axis=0))
        P = D @ P_zero_diagonal
        P_powered = np.linalg.matrix_power(P,diffusion_powers)
    else:
        P_powered = np.linalg.matrix_power(P,diffusion_powers)
    # take the diffusion probs of the neighborhood
    near_neighbors_only = P_powered * P_thresholded
    laziness_aggregate = np.sum(near_neighbors_only,axis=1)
    if avg_transition_probability:
        ones_matrix = np.ones_like(P_thresholded)
        ones_remaining = ones_matrix * P_thresholded
        local_density = np.sum(ones_remaining,axis=1)
        if verbose: print("local density",local_density)
        # divide by the number of neighbors diffused to
        # TODO: In case of isolated points, replace local density of 0 with 1. THe laziness will evaluate to zero.
        local_density[local_density==0]=1
        laziness_aggregate = laziness_aggregate / local_density
    laziness = laziness_aggregate
    if smoothing: # TODO there are probably more intelligent ways to do this smoothing
        # Local averaging to counter the effects local density
        if verbose: print("Applying smoothing...")
        smoothing_P_powered = np.linalg.matrix_power(P,smoothing)
        average_laziness = smoothing_P_powered @ laziness_aggregate[:,None]
        average_laziness = average_laziness.squeeze()
        laziness = average_laziness
    if return_density:
        # compute sums of neighbors taken into consideration
        ones_matrix = np.ones_like(P_thresholded)
        ones_remaining = ones_matrix * P_thresholded
        local_density = np.sum(ones_remaining,axis=1)
        return laziness, local_density
    return laziness


if __name__ == '__main__':
  X = np.loadtxt(args.file, dtype=float)
  P = DiffusionMatrix(X,kernel_type=args.kernel,k=args.k,sigma=args.sigma)
  # compute laziness values
  ls = curvature(P,diffusion_powers=args.t,aperture=args.aperture)
  print(ls)
  np.savetxt(args.file.replace('.txt','')+'-curvatures.txt',ls)
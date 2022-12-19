'''
Compute optimal transport for input embeddings, using manifold distance (Geodesic)

'''
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import ot

def comp_geodesic(X, labels, n_classes, data_df, k=20, diffusion_powers=8):
    '''
        X: (N, D)
    '''
    # Compute Geodesic
    N,D = X.shape

    # Compute Euclidean Distance (N , N) between pts (N, D)
    dist = scipy.spatial.distance_matrix(X, X)
    print('Compute Euclidean Distance... ', dist.shape)
    
    # Create binary Adj Matrix
    topk = np.sort(dist, axis=1)[:, k].reshape(N, 1) # (N, 1)
    filter_m = np.tile(topk, (1, N)) # (n, n)
    adj_mat = (dist <= filter_m) * 1 # zero out non-neighbors
    print('Create binary Adj Matrix... ', np.mean(np.sum(adj_mat, axis=1)))
    
    # Create Graph and compute shortest path
    graph = nx.from_numpy_matrix(adj_mat)
    path_lens = dict(nx.shortest_path_length(graph))
    print('Compute shortest path... ')
    # print(graph[0])
    
    # geodesci mat from path lengths dict
    geodesic = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            geodesic[i, j] = int(path_lens[i][j])
    print('Geodesci finished ... ', geodesic.shape)

    #

    return geodesic

def comp_op(geodesic):
    '''
    geodesic: (N,N) cost matrix
    '''
    # a and b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T = ot.emd(a, b, geodesic)  # exact linear program
    total_cost = np.sum(T*geodesic)


if __name__ == '__main__':
    X = np.random.rand(100,50) # N x dim

    curvature, class_stats = comp_geodesic(X, [], [], [])

    data_df['curvature'] = curvature
    print(np.mean(curvature))
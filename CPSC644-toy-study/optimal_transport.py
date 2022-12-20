'''
Compute optimal transport for input embeddings, using manifold distance (Geodesic)

'''
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import ot
from glob import glob


def comp_geodesic(X, k=20):
    '''
        X: (N, D)
        geodesic: (N, N)
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

    return geodesic

def comp_op(a, b, geodesic):
    '''
    geodesic: (N,n) cost matrix
    '''
    # a and b are 1D histograms (sum to 1 and positive)
    # M is the ground cost matrix
    T = ot.emd(a, b, geodesic)  # exact linear program
    total_cost = np.sum(T*geodesic)

    return total_cost


def comp_opcost(X, labels, n_classes, geodesic):
    N,D = X.shape
    opc = np.zeros((N,N))

    # Compute op cost marix
    for si in range(n_classes):
        sindex = list(np.squeeze(np.argwhere(labels==si)))
        for ti in range(n_classes):
            tindex = list(np.squeeze(np.argwhere(labels==ti)))
            cost_mat = geodesic[sindex,:][:, tindex]
            cost = comp_op(1/len(sindex), 1/len(tindex), cost_mat)

            opc[si,ti] = cost

    return opc

if __name__ == '__main__':
    files = sorted(glob('./results/embeddings/mnist-NA-val_acc_70%/*'))
    batch_0_file = np.load(files[0])

    image = batch_0_file['image']
    label = batch_0_file['label_true']
    embeddings = batch_0_file['embedding']

    print(image.shape)
    print(label.shape)
    print(embeddings.shape)

    geodesic = comp_geodesic(embeddings, k=20)
    opc = comp_opcost(embeddings, label, n_classes=np.max(label)+1, geodesic=geodesic)

    print(opc.shape)
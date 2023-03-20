'''
Compute optimal transport for input embeddings, using manifold distance (Geodesic)
OR Diffsion EMD

'''
import argparse
import os
import sys
from glob import glob

import numpy as np
import phate
import scprep
import yaml
import ot
from matplotlib import pyplot as plt
from scipy import sparse
import scipy
from tqdm import tqdm
import seaborn as sns
import pygsp

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from log_utils import log
from path_utils import update_config_dirs

from DiffusionEMD import DiffusionCheb


def comp_geodesic(X, k=20):
    '''
        X: (N, D)
        geodesic: (N, N)
    '''
    # Compute Geodesic
    N, D = X.shape

    # Compute Euclidean Distance (N, N) between pts (N, D)
    dist = scipy.spatial.distance_matrix(X, X)
    print('Compute Euclidean Distance... ', dist.shape)

    # Create binary Adj Matrix
    topk = np.sort(dist, axis=1)[:, k].reshape(N, 1)  # (N, 1)
    filter_m = np.tile(topk, (1, N))  # (N, N)
    adj_mat = (dist <= filter_m) * 1  # zero out non-neighbors
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
    total_cost = np.sum(T * geodesic)

    return total_cost


def comp_opcost(X, labels, n_classes, geodesic):
    N, D = X.shape
    opc = np.zeros((n_classes, n_classes))

    # Compute op cost marix
    for si in range(n_classes):
        # NOTE: I modified the following line!
        # ORIGINAL: sindex = list(np.squeeze(np.argwhere(labels == si)))
        sindex = list(np.argwhere(labels == si)[:, 0])
        for ti in range(n_classes):
            # NOTE: I modified the following line!
            # ORIGINAL: tindex = list(np.squeeze(np.argwhere(labels == ti)))
            tindex = list(np.argwhere(labels == ti)[:, 0])
            cost_mat = geodesic[sindex, :][:, tindex]
            a = np.array([1 / len(sindex)] * len(sindex))
            b = np.array([1 / len(tindex)] * len(tindex))
            cost = comp_op(a, b, cost_mat)

            opc[si, ti] = cost

    return opc

def indicator_distribution(labels, n_classes):
    '''
        dists: N x M, N is the common graph node num, M is dist num
    '''
    N = labels.shape[0]
    M = n_classes

    d = np.zeros((N,1))
    dists = []

    for i in range(n_classes):
        ids = list(np.argwhere(labels == i)[:, 0])
        d[ids] = 1
        dists.append(d)
    
    dists = np.concatenate(dists, axis=1)

    return dists



def common_graph(X, k=30, th=1e-3):
    '''
        X: (N, D)
        adj: (N, N)
    '''
    # Compute Adj from affinity matrix
    N, D = X.shape

    # Compute Euclidean Distance (N, N) between pts (N, D)
    dist = scipy.spatial.distance_matrix(X, X)
    print('Compute Euclidean Distance... ', dist.shape)
    #print('dist: ', dist[0,:10])
    eps = np.var(dist)
    gk = np.exp(-dist**2/eps) # Gaussian Kernel
    #print(np.diagonal(gk), gk[0,:10])
    adj = (gk > th) * 1 # TODO: ? 
    #print(np.diagonal(adj), adj[0,:10])
    print('Create binary Adj Matrix... avg row sum: ', np.mean(np.sum(adj, axis=1)))
    
    return adj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--method',
                        help='iso|phate|emd',
                        default='phate')
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))

    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-*' %
             (config.output_save_path, config.dataset, config.contrastive)))

    save_root = './optimal_transport/'
    os.makedirs(save_root, exist_ok=True)
    save_path = '%s/opt-%s-%s.png' % (save_root, config.dataset,
                                        config.contrastive)
    log_path = '%s/opt-%s-%s.txt' % (save_root, config.dataset,
                                       config.contrastive)

    num_rows = len(embedding_folders)
    fig = plt.figure(figsize=(10, 5 * num_rows))

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))
        print(embedding_folder)

        labels, embeddings = None, None

        for file in tqdm(files[:]):
            np_file = np.load(file)
            curr_label = np_file['label_true']
            curr_embedding = np_file['embedding']

            if labels is None:
                labels = curr_label[:, None]  # expand dim to [B, 1]
                embeddings = curr_embedding
            else:
                labels = np.vstack((labels, curr_label[:, None]))
                embeddings = np.vstack((embeddings, curr_embedding))

        N, D = embeddings.shape

        assert labels.shape[0] == N
        assert labels.shape[1] == 1

        if args.method == 'emd':
            #adj = common_graph(embeddings) # N x N
            G =  pygsp.graphs.NNGraph(
                embeddings, epsilon=2, NNtype="radius", rescale=True, center=False
            )
            print('G... avg row sum: ', np.mean(np.sum(G.W, axis=1)))

            dists = indicator_distribution(labels, n_classes=np.max(labels) + 1)

            dc = DiffusionCheb()

            # Embeddings where the L1 distance approximates the Earth Mover's Distance
            #dist_embeddings = dc.fit_transform(adj, dists) # Shape: (10, Ks)
            dist_embeddings = dc.fit_transform(G.W, dists) # Shape: (10, Ks)
            print('EMD embeddings.shape: ', dist_embeddings.shape)

            opc = scipy.spatial.distance_matrix(dist_embeddings,dist_embeddings,p=1)
            #print('opc: ', opc[0, :10])

        
        if args.method == 'iso':
            geodesic = comp_geodesic(embeddings, k=20)
            opc = comp_opcost(embeddings,
                            labels,
                            n_classes=np.max(labels) + 1,
                            geodesic=geodesic)
        elif args.method == 'phate':
            # PHATE dimensionality reduction.
            phate_op = phate.PHATE(knn=10,
                               n_jobs=1,
                               n_components=2,
                               verbose=False)
            phate_op.fit(embeddings)
            diff_pot = phate_op.diff_potential # N x landmark
            geodesic = scipy.spatial.distance.cdist(diff_pot, diff_pot) # NxN
	        
            print(geodesic.shape)

            opc = comp_opcost(embeddings,
                                labels,
                                n_classes=np.max(labels) + 1,
                                geodesic=geodesic)
        print(opc.shape)
        
        ax = fig.add_subplot(num_rows, 1, i + 1)
        sns.heatmap(opc, ax=ax)
        ax.set_title('%s method:%s' %
                     (os.path.basename(embedding_folder), args.method))
        
        fig.tight_layout()
        fig.savefig(save_path)


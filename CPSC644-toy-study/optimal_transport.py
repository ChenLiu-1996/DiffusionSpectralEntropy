'''
Compute optimal transport for input embeddings, using manifold distance (Geodesic)

'''
import argparse
import os
import sys
from glob import glob
import matplotlib.pyplot as plt


import networkx as nx
import numpy as np
import ot
import scipy
import yaml
from tqdm import tqdm
from train_infer import update_config_dirs

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config = update_config_dirs(AttributeHashmap(config))

    embedding_root = '%s/embeddings/%s-%s-' % (
        config.output_save_path, config.dataset, config.contrastive)

    for acc_level in ['val_acc_50%', 'val_acc_70%', 'best_val_acc']:
        embedding_folder = embedding_root + acc_level
        files = sorted(glob(embedding_folder + '/*'))

        labels, embeddings = None, None

        # NOTE: only loading first 2 batches right now.
        # for file in tqdm(files):
        for file in tqdm(files[0:2]):
            np_file = np.load(file)
            curr_label = np_file['label_true']
            curr_embedding = np_file['embedding']

            if labels is None:
                labels = curr_label[:, None]  # expand dim to [B, 1]
                embeddings = curr_embedding
            else:
                labels = np.vstack((labels, curr_label[:, None]))
                embeddings = np.vstack((embeddings, curr_embedding))

        geodesic = comp_geodesic(embeddings, k=20)

        opc = comp_opcost(embeddings,
                          labels,
                          n_classes=np.max(labels) + 1,
                          geodesic=geodesic)
        print(opc.shape)

        np_save_path = '%s_op.npy' % (embedding_folder)
        np.save(np_save_path, opc)

        plt.matshow(opc)
        plt.savefig('%s_op_mat.png' % (embedding_folder))
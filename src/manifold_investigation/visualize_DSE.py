import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn import datasets
from tqdm import tqdm
import magic
import scprep

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from information import exact_eigvals, von_neumann_entropy, shannon_entropy
from diffusion import compute_diffusion_matrix

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/api/')
from dse import diffusion_spectral_entropy
from dsmi import diffusion_spectral_mutual_information

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', action='store_false')
    parser.add_argument('--seed', default=20)

    args = vars(parser.parse_args())
    args = AttributeHashmap(args)

    save_root = './results_toy_data/'
    os.makedirs(save_root, exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 15

    num_classes = 3
    cluster_std = 1
    cluster_std_lists = [1.0, 1.0, 1.0]
    center_box = (-20.0, 20.0)

    N = 3000
    save_path_fig = '%s/visualize_intuitive_DSE.png' % (save_root)

    D_list = [2] # Visualize in dimension 2, 3
    #t_list = [1, 2, 3, 4, 5, 7, 10, 100, 1000]
    t_list = [1, 3, 5, 10, 100, 500, 1000, 5000, 10000]
    strech_factor = 3

    separated_DSEs = [[[] for _ in range(len(t_list))]
                       for _ in range(len(D_list))]
    single_DSEs = [[[] for _ in range(len(t_list))]
                    for _ in range(len(D_list))]
    
    fig = plt.figure(figsize=(3 * len(t_list), 6))
    gs = GridSpec(2, len(t_list), figure=fig)

    # Multiple blobs
    for d, dim in enumerate(D_list):
        embeddings, labels = datasets.make_blobs(
            n_samples=N,
            n_features=dim,
            centers=num_classes,
            cluster_std=cluster_std_lists,
            center_box=center_box,
            random_state=args.seed)
        labels = labels.reshape(N, 1)
        if args.transform == True:
            # transformation = np.random.normal(loc=0,
            #                                     scale=0.3,
            #                                     size=(dim, dim))  # D x D
            transformation = np.diag(np.ones(dim))
            transformation[0,0] = strech_factor
            embeddings = np.dot(embeddings, transformation)
        #embeddings /= np.sqrt(dim)

        '''
            Visualize ^t using MAGIC
        '''
        for j, t in tqdm(enumerate(t_list)):
            embeddings_magic = magic.MAGIC(knn=5, t=t, n_pca=100).fit_transform(embeddings)

            if dim == 3:
                ax = fig.add_subplot(gs[0, j], projection='3d')
            else:
                ax = fig.add_subplot(gs[0, j])
            
            ax.set_ylim(-0, 30)
            ax.set_xlim(-100, 100)
            scprep.plot.scatter2d(embeddings_magic,
                        c=labels,
                        legend=False,
                        title='t=%d'%(t),
                        ax=ax,
                        xticks=True,
                        yticks=True,
                        label_prefix='MAGIC',
                        fontsize=10,
                        s=3)
        
    # Single blob
    for d, dim in enumerate(D_list):
        embeddings, _ = datasets.make_blobs(n_samples=N,
                                            n_features=dim,
                                            centers=1,
                                            cluster_std=cluster_std,
                                            random_state=args.seed)
        if args.transform == True:
            transformation = np.diag(np.ones(dim))
            transformation[0,0] = strech_factor
            embeddings = np.dot(embeddings, transformation)
        #embeddings /= np.sqrt(dim)

        num_per_class = N // num_classes
        labels = np.zeros(N)
        # Randomly assign class labels
        for i in range(1, num_classes + 1):
            inds = np.random.choice(np.where(labels == 0)[0],
                                    num_per_class,
                                    replace=False)
            labels[inds] = i
        labels = labels - 1  # class 1,2,3 -> class 0,1,2
        labels = labels.reshape(N, 1)

        '''
            Visualize ^t using MAGIC
        '''
        for j, t in tqdm(enumerate(t_list)):
            embeddings_magic = magic.MAGIC(knn=5, t=t, n_pca=100).fit_transform(embeddings)

            if dim == 3:
                ax = fig.add_subplot(gs[1, j], projection='3d')
            else:
                ax = fig.add_subplot(gs[1, j])
            
            ax.set_ylim(-10, 20)
            ax.set_xlim(-30, 30)
            scprep.plot.scatter2d(embeddings_magic,
                                    c=labels,
                                    legend=False,
                                    title='t=%d'%(t),
                                    ax=ax,
                                    xticks=True,
                                    yticks=True,
                                    label_prefix='MAGIC',
                                    fontsize=10,
                                    s=3)

    fig.tight_layout()
    fig.savefig(save_path_fig)







    

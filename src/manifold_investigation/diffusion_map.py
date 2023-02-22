import argparse
import os
import sys
from glob import glob

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import numpy as np
import phate
import scprep
import yaml
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from attribute_hashmap import AttributeHashmap
from path_utils import update_config_dirs
from log_utils import log

cifar10_int2name = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

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

    embedding_folders = sorted(
        glob('%s/embeddings/%s-%s-*' %
             (config.output_save_path, config.dataset, config.contrastive)))

    save_root = './diffusion_PHATE/'
    os.makedirs(save_root, exist_ok=True)
    save_path = '%s/PHATE-%s-%s.png' % (save_root, config.dataset,
                                        config.contrastive)
    log_path = '%s/PHATE-%s-%s.txt' % (save_root, config.dataset,
                                        config.contrastive)

    num_rows = len(embedding_folders)
    fig = plt.figure(figsize=(10, 5 * num_rows))

    for i, embedding_folder in enumerate(embedding_folders):
        files = sorted(glob(embedding_folder + '/*'))

        labels, embeddings = None, None

        for file in tqdm(files):
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

        if config.dataset == 'cifar10':
            labels_updated = np.zeros(labels.shape, dtype='object')
            for k in range(N):
                labels_updated[k] = cifar10_int2name[labels[k].item()]
            labels = labels_updated
            del labels_updated

        # PHATE plot.
        # ax = fig.add_subplot(num_rows, 1, 2 * i + 1)
        phate_op = phate.PHATE(random_state=0,
                               n_jobs=1,
                               n_components=2,
                               verbose=False).fit(embeddings)
        data_phate = phate_op.transform()

        #data_phate = phate_op.fit_transform(embeddings)

        # scprep.plot.scatter2d(data_phate,
        #                       c=labels,
        #                       legend_anchor=(1, 1),
        #                       ax=ax,
        #                       title=os.path.basename(embedding_folder),
        #                       xticks=False,
        #                       yticks=False,
        #                       label_prefix='PHATE',
        #                       fontsize=10,
        #                       s=3)
        
        # Diffusion map p_t
        p = phate_op.graph.diff_op.toarray()
        t = phate_op._find_optimal_t(t_max=100, plot=False, ax=None)
        print(p.shape, p.dtype, t.dtype)
        print(p[0, :10])
        p_t = np.linalg.matrix_power(p, t)

        W, V = np.linalg.eig(p_t)
        eigenstr = '%s Eigenvalues: ' % os.path.basename(embedding_folder)
        percentiles = [50,90,95,99]
        for per in percentiles:
            eigenstr += '%.2f percentile: %.7f\t' % (per, np.percentile(W, per))
            eigenstr += '> count: %d; \n' % (W > np.percentile(W, per)).sum()
        
        # Top K eigenvalues
        sorted_idx = np.argsort(W)[::-1]
        W = W[sorted_idx]
        V = V[:, sorted_idx]
        k = 12
        eigenstr += 'Top %d eigenvalues: ' % k
        for kindex in range(k):
            eigenstr += '%.7f ' % W[kindex]

        log(eigenstr, log_path)

        # Diffusion Map Embedding
        diff_embed = V@np.diag((W**0.5))
        print(diff_embed.shape)
        min_inds = np.argmin(diff_embed, 0)[:k]
        max_inds = np.argmax(diff_embed, 0)[:k]

        colors = np.zeros(N)
        colors[min_inds] = 1
        colors[max_inds] = 2

        ax = fig.add_subplot(num_rows, 1, 2 * i + 1)
        scprep.plot.scatter2d(data_phate,
                        c=colors,
                        legend_anchor=(1, 1),
                        ax=ax,
                        title=os.path.basename(embedding_folder),
                        xticks=False,
                        yticks=False,
                        label_prefix='PHATE',
                        fontsize=10,
                        s=3)


        fig.tight_layout()
        fig.savefig(save_path)

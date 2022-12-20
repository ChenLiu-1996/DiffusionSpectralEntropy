'''
Compute local curvature using diffusion curvature for input embeddings

'''
import argparse
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import yaml
from diffusion_curvature.core import DiffusionMatrix
from diffusion_curvature.laziness import curvature
from tqdm import tqdm
from train_infer import update_config_dirs

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/src/utils/')
from attribute_hashmap import AttributeHashmap


def comp_curvature(X, n_classes, data_df, k=20, diffusion_powers=8):
    class_stats_list = []
    global_stats = [0,0] # global mean, global std

    # Diffusion Matrix
    P = DiffusionMatrix(X, kernel_type="adaptive anisotropic", k=k)

    # Curvature
    diffusion_curvatures = curvature(P, diffusion_powers=diffusion_powers)
    data_df['curvature'] = diffusion_curvatures

    # Compute curvature related stats per class
    for ci in range(n_classes):
        c_rows = data_df.loc[data_df['class'] == ci]
        curvs = c_rows['curvature']

        # Append class stats
        class_stats_list.append([np.mean(curvs), np.std(curvs), ci])
        global_stats[0] = global_stats[0] + np.mean(curvs)
        global_stats[1] = global_stats[1] + np.std(curvs)

    class_stats_list.append([global_stats[0]/n_classes, global_stats[1]/n_classes, global_stats[1]]) # global stats
    class_stats_df = pd.DataFrame(class_stats_list,
                                  columns=['mean', 'std', 'class'])
    return class_stats_df


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
        df = pd.DataFrame(embeddings, columns=[f'd{i}' for i in range(0, D)])
        df['class'] = labels

        class_stats_df = comp_curvature(embeddings,
                                        n_classes=np.max(labels) + 1,
                                        data_df=df)

        csv_path = '%s_curvature_stats.csv' % (embedding_folder)
        class_stats_df.to_csv(csv_path, index=False)
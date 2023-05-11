import argparse
import os
import sys
from glob import glob

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
sys.path.insert(0, import_dir + '/embedding_preparation')
from attribute_hashmap import AttributeHashmap
from seed import seed_everything

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
    parser.add_argument('--knn', help='k for knn graph.', type=int, default=10)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    seed_everything(1)

    data_path_list = sorted(
        glob('./results_diffusion_entropy_top100/numpy_files/figure-data-*'))

    data_hashmap = {}

    for data_path in data_path_list:
        dataset_name = data_path.split('figure-data-')[1].split('-')[0]
        method_name = data_path.split(dataset_name + '-')[1].split('-')[0]
        network_name = data_path.split(method_name + '-')[1].split('-')[0]
        seed_name = data_path.split(network_name + '-')[1].split('-')[0]

        method_name = 'supervised' if method_name == 'NA' else method_name

        data_numpy = np.load(data_path)
        data_hashmap['-'.join(
            (dataset_name, method_name, network_name, seed_name))] = {
                'epoch': data_numpy['epoch'],
                'acc': data_numpy['acc'],
                'vne': data_numpy['vne'],
                'mi_Y_sample': data_numpy['mi_Y_sample'],
                # 'mi_input': data_numpy['mi_input'],
            }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    save_path_fig_vne = './main_figure_VNE.png'
    save_path_fig_mi = './main_figure_MI.png'

    # Plot of Diffusion Entropy vs. epoch.
    fig_vne = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_vne)

    for method in ['supervised', 'simclr', 'wronglabel']:
        for dataset in ['mnist', 'cifar10']:
            for seed in [1, 2, 3]:
                string = '%s-%s-resnet50-seed%s' % (dataset, method, seed)
                if string not in data_hashmap.keys():
                    data_hashmap[string] = {
                        'epoch': [],
                        'acc': [],
                        'vne': [],
                        'mi_Y_sample': [],
                    }

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_vne.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['vne'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['vne'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['vne'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised', fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('SimCLR', fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting', fontsize=25)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # if dataset == 'mnist':
            #     ax.set_ylim([6, 14])
            # else:
            #     ax.set_ylim([0, 15])

            ax = fig_vne.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['vne'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['vne'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['vne'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # if dataset == 'mnist':
            #     ax.set_ylim([6, 14])
            # else:
            #     ax.set_ylim([0, 15])

    fig_vne.tight_layout()
    fig_vne.savefig(save_path_fig_vne)
    plt.close(fig=fig_vne)

    # Plot of Mutual Information vs. epoch.
    fig_mi = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_mi)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_mi.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['mi_Y_sample'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['mi_Y_sample'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['mi_Y_sample'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised', fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('SimCLR', fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting', fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # if dataset == 'mnist':
            #     ax.set_ylim([6, 14])
            # else:
            #     ax.set_ylim([0, 15])

            ax = fig_mi.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['mi_Y_sample'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['mi_Y_sample'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['mi_Y_sample'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            # if dataset == 'mnist':
            #     ax.set_ylim([6, 14])
            # else:
            #     ax.set_ylim([0, 15])

    fig_mi.tight_layout()
    fig_mi.savefig(save_path_fig_mi)
    plt.close(fig=fig_mi)
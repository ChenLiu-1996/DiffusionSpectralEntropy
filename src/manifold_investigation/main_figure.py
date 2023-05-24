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
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--zero-init', action='store_true')
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    seed_everything(1)

    data_path_list_mnist = sorted(
        glob(
            './results_diffusion_entropy_t=1/numpy_files/figure-data-*mnist*'))
    data_path_list_cifar10 = sorted(
        glob(
            './results_diffusion_entropy_t=2/numpy_files/figure-data-*cifar10*'
        ))

    data_path_list = [item for item in data_path_list_mnist
                      ] + [item for item in data_path_list_cifar10]

    data_hashmap = {}

    for data_path in data_path_list:
        dataset_name = data_path.split('figure-data-')[1].split('-')[0]
        method_name = data_path.split(dataset_name + '-')[1].split('-')[0]
        network_name = data_path.split(method_name + '-')[1].split('-')[0]
        seed_name = data_path.split(network_name + '-')[1].split('.npy')[0]

        method_name = 'supervised' if method_name == 'NA' else method_name

        data_numpy = np.load(data_path)
        data_hashmap['-'.join(
            (dataset_name, method_name, network_name, seed_name))] = {
                'epoch': data_numpy['epoch'],
                'acc': data_numpy['acc'],
                'se': data_numpy['se'],
                'vne': data_numpy['vne'],
                'mi_Y': data_numpy['mi_Y'],
                'mi_X': data_numpy['mi_X'],
                'mi_Y_shannon': data_numpy['mi_Y_shannon'],
                'mi_X_shannon': data_numpy['mi_X_shannon'],
                'H_ZgivenY': data_numpy['H_ZgivenY'],
            }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    save_root = './main_figures/'
    os.makedirs(save_root, exist_ok=True)

    # NOTE: Classic Shannon Entropy
    save_path_fig_CSE = save_root + 'main_figure_CSE.png'
    # NOTE: Diffusion Spectral Entropy
    save_path_fig_DSE = save_root + 'main_figure_DSE.png'
    # NOTE: Diffusion Spectral Mutual Information w.r.t. output
    save_path_fig_DSMI_Y = save_root + 'main_figure_DSMI_Y.png'
    # NOTE: Diffusion Spectral Mutual Information w.r.t. input
    save_path_fig_DSMI_X = save_root + 'main_figure_DSMI_X.png'
    # NOTE: Classic Shannon Mutual Information w.r.t. output
    save_path_fig_CSMI_Y = save_root + 'main_figure_CSMI_Y.png'
    # NOTE: Classic Shannon Mutual Information w.r.t. input
    save_path_fig_CSMI_X = save_root + 'main_figure_CSMI_X.png'

    save_path_fig_H_ZgivenY = save_root + 'main_figure_H_ZgivenY.png'

    for method in ['supervised', 'simclr', 'wronglabel']:
        for dataset in ['mnist', 'cifar10']:
            for seed in [1, 2, 3]:
                string = '%s-%s-resnet50-seed%s' % (dataset, method, seed)
                if string not in data_hashmap.keys():
                    data_hashmap[string] = {
                        'epoch': [],
                        'acc': [],
                        'se': [],
                        'vne': [],
                        'mi_Y': [],
                        'mi_X': [],
                        'mi_Y_shannon': [],
                        'mi_X_shannon': [],
                        'H_ZgivenY': [],
                    }

    # Plot of Diffusion Entropy vs. epoch.
    fig_CSE = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_CSE)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_CSE.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['se'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['se'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['se'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\nCSE', fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nCSE', fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nCSE', fontsize=25)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_CSE.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['se'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['se'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['se'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fig_CSE.tight_layout()
    fig_CSE.savefig(save_path_fig_CSE)
    plt.close(fig=fig_CSE)

    # Plot of Diffusion Entropy vs. epoch.
    fig_DSE = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_DSE)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_DSE.add_subplot(gs[gs_x, gs_y * 2])
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
                ax.set_ylabel('Supervised\nDSE ' + r'$S_D(Z)$', fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nDSE ' + r'$S_D(Z)$', fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nDSE ' + r'$S_D(Z)$', fontsize=25)

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

            ax = fig_DSE.add_subplot(gs[gs_x, gs_y * 2 + 1])
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

    fig_DSE.tight_layout()
    fig_DSE.savefig(save_path_fig_DSE)
    plt.close(fig=fig_DSE)

    # Plot of I(Z; Y)
    fig_DSMI_Y = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_DSMI_Y)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_DSMI_Y.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['mi_Y'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['mi_Y'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['mi_Y'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\nDSMI ' + r'$I_D(Z; Y)$',
                              fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nDSMI ' + r'$I_D(Z; Y)$',
                              fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nDSMI ' + r'$I_D(Z; Y)$',
                              fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_DSMI_Y.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['mi_Y'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['mi_Y'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['mi_Y'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_DSMI_Y.tight_layout()
    fig_DSMI_Y.savefig(save_path_fig_DSMI_Y)
    plt.close(fig=fig_DSMI_Y)

    # Plot of I(Z; Y), Shannon
    fig_CSMI_Y = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_CSMI_Y)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_CSMI_Y.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['mi_Y_shannon'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['mi_Y_shannon'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['mi_Y_shannon'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\nCSMI ' + r'$I(Z; Y)$',
                              fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nCSMI ' + r'$I(Z; Y)$',
                              fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nCSMI ' + r'$I(Z; Y)$',
                              fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_CSMI_Y.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['mi_Y_shannon'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['mi_Y_shannon'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['mi_Y_shannon'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_CSMI_Y.tight_layout()
    fig_CSMI_Y.savefig(save_path_fig_CSMI_Y)
    plt.close(fig=fig_CSMI_Y)

    # Plot of I(Z; X)
    fig_DSMI_X = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_DSMI_X)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_DSMI_X.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['mi_X'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['mi_X'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['mi_X'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\nDSMI ' + r'$I_D(Z; X)$',
                              fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nDSMI ' + r'$I_D(Z; X)$',
                              fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nDSMI ' + r'$I_D(Z; X)$',
                              fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_DSMI_X.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['mi_X'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['mi_X'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['mi_X'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_DSMI_X.tight_layout()
    fig_DSMI_X.savefig(save_path_fig_DSMI_X)
    plt.close(fig=fig_DSMI_X)

    # Plot of I(Z; X), Shannon
    fig_CSMI_X = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_CSMI_X)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_CSMI_X.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['mi_X_shannon'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['mi_X_shannon'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['mi_X_shannon'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\nCSMI ' + r'$I(Z; X)$',
                              fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\nCSMI ' + r'$I(Z; X)$',
                              fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\nCSMI ' + r'$I(Z; X)$',
                              fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_CSMI_X.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['mi_X_shannon'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['mi_X_shannon'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['mi_X_shannon'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_CSMI_X.tight_layout()
    fig_CSMI_X.savefig(save_path_fig_CSMI_X)
    plt.close(fig=fig_CSMI_X)

    # Plot H(Z|Y)
    fig_H_ZgivenY = plt.figure(figsize=(30, 12))
    gs = GridSpec(3, 4, figure=fig_H_ZgivenY)

    color_map = ['mediumblue', 'darkred', 'darkgreen']
    for method, gs_x in zip(['supervised', 'simclr', 'wronglabel'], [0, 1, 2]):
        for dataset, gs_y in zip(['mnist', 'cifar10'], [0, 1]):
            ax = fig_H_ZgivenY.add_subplot(gs[gs_x, gs_y * 2])
            ax.spines[['right', 'top']].set_visible(False)
            ax.plot(data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed1' %
                                 (dataset, method)]['H_ZgivenY'],
                    color=color_map[0],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed2' %
                                 (dataset, method)]['H_ZgivenY'],
                    color=color_map[1],
                    linewidth=3,
                    alpha=0.5)
            ax.plot(data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['epoch'],
                    data_hashmap['%s-%s-resnet50-seed3' %
                                 (dataset, method)]['H_ZgivenY'],
                    color=color_map[2],
                    linewidth=3,
                    alpha=0.5)
            if gs_x == 0 and gs_y == 0:
                ax.set_ylabel('Supervised\n' + r'$H(Z | Y)$', fontsize=25)
            if gs_x == 1 and gs_y == 0:
                ax.set_ylabel('Contrastive\n' + r'$H(Z | Y)$', fontsize=25)
            if gs_x == 2 and gs_y == 0:
                ax.set_ylabel('Overfitting\n' + r'$H(Z | Y)$', fontsize=25)
            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Epochs Trained', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            ax = fig_H_ZgivenY.add_subplot(gs[gs_x, gs_y * 2 + 1])
            ax.spines[['right', 'top']].set_visible(False)

            ax.scatter(data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed1' %
                                    (dataset, method)]['H_ZgivenY'],
                       color=color_map[0],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed2' %
                                    (dataset, method)]['H_ZgivenY'],
                       color=color_map[1],
                       alpha=0.2)
            ax.scatter(data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['acc'],
                       data_hashmap['%s-%s-resnet50-seed3' %
                                    (dataset, method)]['H_ZgivenY'],
                       color=color_map[2],
                       alpha=0.2)

            if gs_x == 0:
                ax.set_title(dataset.upper(), fontsize=25)
            if gs_x == 2:
                ax.set_xlabel('Downstream Accuracy', fontsize=25)
            ax.tick_params(axis='both', which='major', labelsize=20)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig_H_ZgivenY.tight_layout()
    fig_H_ZgivenY.savefig(save_path_fig_H_ZgivenY)
    plt.close(fig=fig_H_ZgivenY)
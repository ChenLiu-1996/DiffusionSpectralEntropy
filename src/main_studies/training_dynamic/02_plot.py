import os
from glob import glob

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


def plot_main_figure(metric: str, fig_save_path: str) -> None:
    dataset_formal_map = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'stl10': 'STL-10',
        'tinyimagenet': 'ImageNet-T',
    }
    method_formal_map = {
        'supervised': 'Supervised',
        'simclr': 'Contrastive',
        'wronglabel': 'Overfitting',
    }
    network_formal_map = {
        'resnet': 'ResNet',
        'resnext': 'ResNeXT',
        'convnext': 'ConvNeXT',
        'vit': 'ViT',
        'swin': 'Swin Trans.',
        'xcit': 'XCiT',
    }
    metric_formal_map = {
        'dse_Z': 'DSE ' + r'$S_D(Z)$',
        'cse_Z': 'CSE ' + r'$H(Z)$',
        'dsmi_Z_Y': 'DSMI ' + r'$I_D(Z; Y)$',
        'dsmi_Z_X': 'DSMI ' + r'$I_D(Z; X)$',
        'csmi_Z_Y': 'CSMI ' + r'$I(Z; Y)$',
        'csmi_Z_X': 'CSMI ' + r'$I(Z; X)$',
    }

    fig = plt.figure(figsize=(6 * len(METHOD_LIST), 10))
    gs = GridSpec(2, len(DATASET_LIST), figure=fig)
    my_palette = sns.color_palette('icefire', n_colors=len(METHOD_LIST))

    for dataset, gs_y in zip(DATASET_LIST, range(len(DATASET_LIST))):
        #NOTE: First find the x and y range.
        all_metric_list = []
        for network in NETWORK_LIST:
            for method_idx, method in enumerate(METHOD_LIST):
                for seed in SEED_LIST:
                    curr_metric_list = data_hashmap['%s-%s-%s-seed%s' %
                                                    (dataset, method, network,
                                                     seed)][metric]

                    if len(curr_metric_list) > 0:
                        all_metric_list.extend(curr_metric_list)
                    del curr_metric_list

        if len(all_metric_list) > 0:
            acc_lim = [0, 100]
            metric_range = np.max(all_metric_list) - np.min(all_metric_list)
            metric_lim = [
                np.min(all_metric_list) - 0.1 * metric_range,
                np.max(all_metric_list) + 0.1 * metric_range
            ]
        else:
            acc_lim = [0, 100]
            metric_lim = [0, 1]
        del all_metric_list

        #NOTE: Now start plotting.
        ax1 = fig.add_subplot(gs[0, gs_y])
        ax1.spines[['right', 'top']].set_visible(False)
        ax1.set_title('ConvNets on %s' % dataset_formal_map[dataset],
                      fontsize=20)
        ax1.set_xlim(acc_lim)
        ax1.set_ylim(metric_lim)
        ax1.set_ylabel(metric_formal_map[metric], fontsize=18)
        ax1.set_xlabel('Val. Accuracy', fontsize=18)
        ax1.grid()
        ax1.tick_params(axis='both', which='major', labelsize=15)
        ax2 = fig.add_subplot(gs[1, gs_y])
        ax2.set_title('ViTs on %s' % dataset_formal_map[dataset], fontsize=20)
        ax2.spines[['right', 'top']].set_visible(False)
        ax2.set_xlim(acc_lim)
        ax2.set_ylim(metric_lim)
        ax2.set_ylabel(metric_formal_map[metric], fontsize=18)
        ax2.set_xlabel('Val. Accuracy', fontsize=18)
        ax2.grid()
        ax2.tick_params(axis='both', which='major', labelsize=15)

        for method_idx, method in enumerate(METHOD_LIST):
            convnets_acc_list, convnets_metric_list = None, None
            vits_acc_list, vits_metric_list = None, None
            for network in NETWORK_LIST:
                x_axis, y_axis = None, None
                for seed in SEED_LIST:
                    curr_acc_list = data_hashmap['%s-%s-%s-seed%s' %
                                                 (dataset, method, network,
                                                  seed)]['val_acc']
                    curr_metric_list = data_hashmap['%s-%s-%s-seed%s' %
                                                    (dataset, method, network,
                                                     seed)][metric]

                    if x_axis is None and len(curr_acc_list) > 0:
                        x_axis = curr_acc_list[:, None]
                        y_axis = curr_metric_list[:, None]
                    else:
                        if len(x_axis) == len(curr_acc_list):
                            x_axis = np.hstack((x_axis, curr_acc_list[:,
                                                                      None]))
                            y_axis = np.hstack(
                                (y_axis, curr_metric_list[:, None]))

                if method != 'simclr':
                    # Subsample for less crowded plot.
                    x_axis = x_axis[::5]
                    y_axis = y_axis[::5]

                if x_axis.shape[-1] != len(SEED_LIST):
                    x_axis = np.repeat(x_axis, repeats=len(SEED_LIST),
                                       axis=-1)[..., :len(SEED_LIST)]
                    assert x_axis.shape[-1] == len(SEED_LIST)
                    y_axis = np.repeat(y_axis, repeats=len(SEED_LIST),
                                       axis=-1)[..., :len(SEED_LIST)]
                    assert y_axis.shape[-1] == len(SEED_LIST)

                if network in NETWORK_LIST_CONV:
                    if convnets_acc_list is None and len(x_axis) > 0:
                        convnets_acc_list = x_axis[..., None]
                        convnets_metric_list = y_axis[..., None]
                    elif len(x_axis) > 0:
                        convnets_acc_list = np.dstack(
                            (convnets_acc_list, x_axis[..., None]))
                        convnets_metric_list = np.dstack(
                            (convnets_metric_list, y_axis[..., None]))
                else:
                    assert network in NETWORK_LIST_VIT
                    if vits_acc_list is None and len(x_axis) > 0:
                        vits_acc_list = x_axis[..., None]
                        vits_metric_list = y_axis[..., None]
                    elif len(x_axis) > 0:
                        vits_acc_list = np.dstack(
                            (vits_acc_list, x_axis[..., None]))
                        vits_metric_list = np.dstack(
                            (vits_metric_list, y_axis[..., None]))

            if convnets_acc_list is not None:
                for network_idx in range(len(NETWORK_LIST_CONV)):
                    for seed_idx in range(len(SEED_LIST)):
                        ax1.scatter(convnets_acc_list[:, seed_idx,
                                                      network_idx],
                                    convnets_metric_list[:, seed_idx,
                                                         network_idx],
                                    color=my_palette[method_idx],
                                    s=100,
                                    alpha=0.2,
                                    marker=LEGEND_MARKER_LIST[network_idx],
                                    label='_nolegend_')
                if gs_y == 0:
                    legend_elements = []
                    for _method_idx, method in enumerate(METHOD_LIST):
                        legend_elements.append(
                            Line2D([0], [0],
                                   marker='o',
                                   color='w',
                                   markerfacecolor=my_palette[_method_idx],
                                   markeredgecolor=my_palette[_method_idx],
                                   alpha=0.2,
                                   label=method_formal_map[method],
                                   markersize=12))
                    for network_idx, network in enumerate(NETWORK_LIST_CONV):
                        legend_elements.append(
                            Line2D([0], [0],
                                   marker=LEGEND_MARKER_LIST[network_idx],
                                   color='w',
                                   markeredgecolor='k',
                                   label=network_formal_map[network],
                                   markersize=12))
                    ax1.legend(handles=legend_elements, fontsize=14, ncol=2)
            if vits_metric_list is not None:
                for network_idx in range(len(NETWORK_LIST_VIT)):
                    for seed_idx in range(len(SEED_LIST)):
                        ax2.scatter(vits_acc_list[:, seed_idx, network_idx],
                                    vits_metric_list[:, seed_idx, network_idx],
                                    color=my_palette[method_idx],
                                    s=100,
                                    alpha=0.2,
                                    marker=LEGEND_MARKER_LIST[network_idx],
                                    label='_nolegend_')
                if gs_y == 0:
                    legend_elements = []
                    for network_idx, network in enumerate(NETWORK_LIST_VIT):
                        legend_elements.append(
                            Line2D([0], [0],
                                   marker=LEGEND_MARKER_LIST[network_idx],
                                   color='w',
                                   markeredgecolor='k',
                                   label=network_formal_map[network],
                                   markersize=12))
                    ax2.legend(handles=legend_elements, fontsize=14)

    fig.tight_layout()
    fig.savefig(fig_save_path)
    plt.close(fig=fig)


def plot_supp_figure(metric: str, fig_save_path: str) -> None:
    dataset_formal_map = {
        'mnist': 'MNIST',
        'cifar10': 'CIFAR-10',
        'stl10': 'STL-10',
        'tinyimagenet': 'ImageNet-T',
    }
    method_formal_map = {
        'supervised': 'Supervised',
        'simclr': 'Contrastive',
        'wronglabel': 'Overfitting',
    }
    network_formal_map = {
        'resnet': 'ResNet',
        'resnext': 'ResNeXT',
        'convnext': 'ConvNeXT',
        'vit': 'ViT',
        'swin': 'Swin Trans.',
        'xcit': 'XCiT',
    }
    metric_formal_map = {
        'dse_Z': 'DSE ' + r'$S_D(Z)$',
        'cse_Z': 'CSE ' + r'$H(Z)$',
        'dsmi_Z_Y': 'DSMI ' + r'$I_D(Z; Y)$',
        'dsmi_Z_X': 'DSMI ' + r'$I_D(Z; X)$',
        'csmi_Z_Y': 'CSMI ' + r'$I(Z; Y)$',
        'csmi_Z_X': 'CSMI ' + r'$I(Z; X)$',
    }

    fig = plt.figure(figsize=(60, 5 * len(METHOD_LIST)))
    gs = GridSpec(len(DATASET_LIST), len(NETWORK_LIST) * 2, figure=fig)
    my_palette = sns.color_palette('icefire', n_colors=len(METHOD_LIST))

    for dataset, gs_x in zip(DATASET_LIST, range(len(DATASET_LIST))):
        #NOTE: First find the x and y range.
        all_epoch_list, all_metric_list = [], []
        for network in NETWORK_LIST:
            for method_idx, method in enumerate(METHOD_LIST):
                for seed in SEED_LIST:
                    curr_epoch_list = data_hashmap['%s-%s-%s-seed%s' %
                                                   (dataset, method, network,
                                                    seed)]['epoch']
                    curr_metric_list = data_hashmap['%s-%s-%s-seed%s' %
                                                    (dataset, method, network,
                                                     seed)][metric]

                    if len(curr_epoch_list) > 0:
                        all_epoch_list.extend(curr_epoch_list)
                    if len(curr_metric_list) > 0:
                        all_metric_list.extend(curr_metric_list)
                    del curr_epoch_list, curr_metric_list

        if len(all_epoch_list) > 0:
            epoch_lim = [np.min(all_epoch_list), np.max(all_epoch_list) + 5]
            acc_lim = [0, 100]
            metric_range = np.max(all_metric_list) - np.min(all_metric_list)
            metric_lim = [
                np.min(all_metric_list) - 0.1 * metric_range,
                np.max(all_metric_list) + 0.1 * metric_range
            ]
        else:
            epoch_lim = [0, 1]
            acc_lim = [0, 100]
            metric_lim = [0, 1]
        del all_epoch_list, all_metric_list

        #NOTE: Now start plotting.
        for network, gs_y in zip(NETWORK_LIST, range(len(NETWORK_LIST))):
            # Odd rows: Metric vs. Training Epoch.
            ax = fig.add_subplot(gs[gs_x, gs_y])
            ax.spines[['right', 'top']].set_visible(False)

            for method_idx, method in enumerate(METHOD_LIST):
                x_axis, y_axis = None, None
                for seed in SEED_LIST:
                    curr_epoch_list = data_hashmap['%s-%s-%s-seed%s' %
                                                   (dataset, method, network,
                                                    seed)]['epoch']
                    curr_metric_list = data_hashmap['%s-%s-%s-seed%s' %
                                                    (dataset, method, network,
                                                     seed)][metric]
                    if x_axis is None and len(curr_epoch_list) > 0:
                        x_axis = curr_epoch_list
                        y_axis = curr_metric_list[:, None]
                    else:
                        if len(x_axis) == len(curr_epoch_list):
                            y_axis = np.hstack(
                                (y_axis, curr_metric_list[:, None]))

                ax.set_xlim(epoch_lim)
                ax.set_ylim(metric_lim)
                if method != 'simclr':
                    # Subsample for less crowded plot.
                    x_axis = x_axis[::5]
                    y_axis = y_axis[::5]

                if y_axis.shape[-1] != len(SEED_LIST):
                    y_axis = np.repeat(y_axis, repeats=len(SEED_LIST),
                                       axis=-1)[..., :len(SEED_LIST)]
                    assert y_axis.shape[-1] == len(SEED_LIST)

                ax.plot(x_axis,
                        np.mean(y_axis, axis=-1),
                        color=my_palette[method_idx],
                        lw=4)
                if y_axis.shape[0] > 1:
                    ax.fill_between(
                        x_axis,
                        np.mean(y_axis, axis=-1) - np.std(y_axis, axis=-1),
                        np.mean(y_axis, axis=-1) + np.std(y_axis, axis=-1),
                        color=my_palette[method_idx],
                        alpha=0.2,
                        label='_nolegend_')
                ax.grid()
                ax.tick_params(axis='both', which='major', labelsize=15)

            if gs_x == 0 and gs_y == 0:
                ax.legend([method_formal_map[item] for item in METHOD_LIST],
                          fontsize=16)
            ax.set_ylabel(metric_formal_map[metric], fontsize=25)
            ax.set_title(
                '%s on \n %s' %
                (network_formal_map[network], dataset_formal_map[dataset]),
                fontsize=25)
            ax.set_xlabel('Epochs Trained', fontsize=25)

            # Even rows: Metric vs. Val Acc.
            ax = fig.add_subplot(gs[gs_x, len(NETWORK_LIST) + gs_y])
            ax.spines[['right', 'top']].set_visible(False)

            for method_idx, method in enumerate(METHOD_LIST):
                x_axis, y_axis = None, None
                for seed in SEED_LIST:
                    curr_acc_list = data_hashmap['%s-%s-%s-seed%s' %
                                                 (dataset, method, network,
                                                  seed)]['val_acc']
                    curr_metric_list = data_hashmap['%s-%s-%s-seed%s' %
                                                    (dataset, method, network,
                                                     seed)][metric]

                    if x_axis is None and len(curr_acc_list) > 0:
                        x_axis = curr_acc_list[:, None]
                        y_axis = curr_metric_list[:, None]
                    else:
                        if len(x_axis) == len(curr_acc_list):
                            x_axis = np.hstack((x_axis, curr_acc_list[:,
                                                                      None]))
                            y_axis = np.hstack(
                                (y_axis, curr_metric_list[:, None]))

                if method != 'simclr':
                    # Subsample for less crowded plot.
                    x_axis = x_axis[::5]
                    y_axis = y_axis[::5]

                if x_axis.shape[-1] != len(SEED_LIST):
                    x_axis = np.repeat(x_axis, repeats=len(SEED_LIST),
                                       axis=-1)[..., :len(SEED_LIST)]
                    assert x_axis.shape[-1] == len(SEED_LIST)
                    y_axis = np.repeat(y_axis, repeats=len(SEED_LIST),
                                       axis=-1)[..., :len(SEED_LIST)]
                    assert y_axis.shape[-1] == len(SEED_LIST)

                ax.set_xlim(acc_lim)
                ax.set_ylim(metric_lim)
                ax.scatter(x_axis,
                           y_axis,
                           color=my_palette[method_idx],
                           s=100,
                           facecolor='none')
                ax.grid()
                ax.tick_params(axis='both', which='major', labelsize=15)

            ax.set_ylabel(metric_formal_map[metric], fontsize=25)
            ax.set_title(
                '%s on \n %s' %
                (network_formal_map[network], dataset_formal_map[dataset]),
                fontsize=25)
            ax.set_xlabel('Val. Accuracy', fontsize=25)

    fig.tight_layout()
    fig.savefig(fig_save_path)
    plt.close(fig=fig)


if __name__ == '__main__':
    DATASET_LIST = ['mnist', 'cifar10', 'stl10']
    SEED_LIST = [1, 2, 3]
    METHOD_LIST = ['supervised', 'simclr', 'wronglabel']
    NETWORK_LIST = ['resnet', 'resnext', 'convnext', 'vit', 'swin', 'xcit']
    NETWORK_LIST_CONV = ['resnet', 'resnext', 'convnext']
    NETWORK_LIST_VIT = ['vit', 'swin', 'xcit']
    LEGEND_MARKER_LIST = ['o', 's', '*']
    METRIC_LIST = [
        'dse_Z', 'cse_Z', 'dsmi_Z_Y', 'csmi_Z_Y', 'dsmi_Z_X', 'csmi_Z_X'
    ]

    data_path_list = []
    for dataset in DATASET_LIST:
        data_path_list_mnist = glob('./results/%s-*/results.npz' % dataset)
        data_path_list.extend(list(data_path_list_mnist))

    data_hashmap = {}
    for data_path in data_path_list:
        dataset_name = data_path.split('./results/')[1].split('-')[0]
        method_name = data_path.split(dataset_name + '-')[1].split('-')[0]
        network_name = data_path.split(method_name + '-')[1].split('-')[0]
        seed_name = data_path.split(network_name + '-')[1].split('/')[0]

        data_numpy = np.load(data_path)
        data_hashmap['-'.join(
            (dataset_name, method_name, network_name, seed_name))] = {
                'epoch': data_numpy['epoch'],
                'val_acc': data_numpy['val_acc'],
                'dse_Z': data_numpy['dse_Z'],
                'cse_Z': data_numpy['cse_Z'],
                'dsmi_Z_X': data_numpy['dsmi_Z_X'],
                'csmi_Z_X': data_numpy['csmi_Z_X'],
                'dsmi_Z_Y': data_numpy['dsmi_Z_Y'],
                'csmi_Z_Y': data_numpy['csmi_Z_Y'],
            }

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['legend.fontsize'] = 20

    save_root = './main_figures/'
    os.makedirs(save_root, exist_ok=True)

    # NOTE: Diffusion Spectral Entropy
    save_path_fig_DSE = save_root + 'main_figure_DSE.png'
    save_path_fig_DSE_supp = save_root + 'supp_figure_DSE.png'
    # NOTE: Classic Shannon Entropy
    save_path_fig_CSE = save_root + 'main_figure_CSE.png'
    save_path_fig_CSE_supp = save_root + 'supp_figure_CSE.png'
    # NOTE: Diffusion Spectral Mutual Information w.r.t. output
    save_path_fig_DSMI_Z_Y = save_root + 'main_figure_DSMI_Z_Y.png'
    save_path_fig_DSMI_Z_Y_supp = save_root + 'supp_figure_DSMI_Z_Y.png'
    # NOTE: Diffusion Spectral Mutual Information w.r.t. input
    save_path_fig_DSMI_Z_X = save_root + 'main_figure_DSMI_Z_X.png'
    save_path_fig_DSMI_Z_X_supp = save_root + 'supp_figure_DSMI_Z_X.png'
    # NOTE: Classic Shannon Mutual Information w.r.t. output
    save_path_fig_CSMI_Z_Y = save_root + 'main_figure_CSMI_Z_Y.png'
    save_path_fig_CSMI_Z_Y_supp = save_root + 'supp_figure_CSMI_Z_Y.png'
    # NOTE: Classic Shannon Mutual Information w.r.t. input
    save_path_fig_CSMI_Z_X = save_root + 'main_figure_CSMI_Z_X.png'
    save_path_fig_CSMI_Z_X_supp = save_root + 'supp_figure_CSMI_Z_X.png'

    for method in METHOD_LIST:
        for dataset in DATASET_LIST:
            for network in NETWORK_LIST:
                for seed in SEED_LIST:
                    string = '%s-%s-%s-seed%s' % (dataset, method, network,
                                                  seed)
                    if string not in data_hashmap.keys():
                        data_hashmap[string] = {
                            'epoch': [],
                            'val_acc': [],
                            'acc_diverg': [],
                            'dse_Z': [],
                            'cse_Z': [],
                            'dsmi_Z_X': [],
                            'csmi_Z_X': [],
                            'dsmi_Z_Y': [],
                            'csmi_Z_Y': [],
                        }

    plot_main_figure(metric='dse_Z', fig_save_path=save_path_fig_DSE)
    plot_main_figure(metric='cse_Z', fig_save_path=save_path_fig_CSE)
    plot_main_figure(metric='dsmi_Z_Y', fig_save_path=save_path_fig_DSMI_Z_Y)
    plot_main_figure(metric='dsmi_Z_X', fig_save_path=save_path_fig_DSMI_Z_X)
    plot_main_figure(metric='csmi_Z_Y', fig_save_path=save_path_fig_CSMI_Z_Y)
    plot_main_figure(metric='csmi_Z_X', fig_save_path=save_path_fig_CSMI_Z_X)

    plot_supp_figure(metric='dse_Z', fig_save_path=save_path_fig_DSE_supp)
    plot_supp_figure(metric='cse_Z', fig_save_path=save_path_fig_CSE_supp)
    plot_supp_figure(metric='dsmi_Z_Y',
                     fig_save_path=save_path_fig_DSMI_Z_Y_supp)
    plot_supp_figure(metric='dsmi_Z_X',
                     fig_save_path=save_path_fig_DSMI_Z_X_supp)
    plot_supp_figure(metric='csmi_Z_Y',
                     fig_save_path=save_path_fig_CSMI_Z_Y_supp)
    plot_supp_figure(metric='csmi_Z_X',
                     fig_save_path=save_path_fig_CSMI_Z_X_supp)

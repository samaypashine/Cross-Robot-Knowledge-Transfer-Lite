# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from paper5.utils import get_config, get_split_data_objects, get_classes_labels, get_dim_reduction_fn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the dataset from binary data.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset'],
                        # required=True,
                        default='Tool_Dataset',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-feature',
                        choices=['discretized-10-bins', 'discretized-20-bins', 'autoencoder-linear',
                                 'autoencoder-linear-tl'],
                        default='discretized-10-bins',
                        help='feature type')
    parser.add_argument('-dim-reduction',
                        choices=['PCA', 'ISOMAP', 'TSNE'],
                        default='PCA',
                        help='dimensionality reduction function')
    args = parser.parse_args()

    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    plot_dataset_path = r'data' + os.sep + args.dataset + '_Plot_Features'

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    logging.info('config: {}'.format(config))

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities = config[args.dataset][args.robot]['modalities']
    logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([binary_dataset_path, args.robot, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()
    logging.info('metadata: {}'.format(metadata))

    objects = metadata[behaviors[0]]['objects']
    tools = metadata[behaviors[0]]['tools']
    trials = metadata[behaviors[0]]['trials']
    logging.info('objects: {}'.format(objects))
    logging.info('tools: {}'.format(tools))
    logging.info('trials: {}'.format(trials))

    dim_reduction_fn = get_dim_reduction_fn(args.dim_reduction)

    object_colors = ['orangered', 'blue', 'darkgreen', 'orange', 'maroon', 'lightblue', 'magenta', 'olive', 'brown',
                     'cyan', 'darkblue', 'beige', 'chartreuse', 'gold', 'green', 'grey', 'coral', 'black', 'khaki',
                     'orchid', 'steelblue', 'chocolate', 'indigo', 'crimson', 'fuchsia']
    tool_markers = ['o', '^', 's', 'P', '*', 'X', 'D']

    if args.dataset == 'Tool_Dataset_2Tools_2Contents':
        if isinstance(args.feature, str):
            bbox_to_anchor_tools = (1.28, 1)
            bbox_to_anchor_objects = (1.235, 0.75)
        else:
            bbox_to_anchor_tools = (1.6, 1)
            bbox_to_anchor_objects = (1.5, 0.7)
    else:
        if isinstance(args.feature, str):
            bbox_to_anchor_tools = (1.34, 1)
            bbox_to_anchor_objects = (1.64, 1)
        else:
            # For 2 types of features
            bbox_to_anchor_tools = (1.74, 1)
            bbox_to_anchor_objects = (2.4, 1)

    for behavior in behaviors:
        logging.info('behavior: {}'.format(behavior))
        for modality in modalities:
            logging.info('modality: {}'.format(modality))

            # Plot features in a separate plot for args.feature
            if isinstance(args.feature, str):

                objects_labels = get_classes_labels(metadata[behaviors[0]]['objects'])
                logging.info('objects_labels: {}'.format(objects_labels))

                tools_labels = get_classes_labels(metadata[behaviors[0]]['tools'])
                logging.info('tools_labels: {}'.format(tools_labels))

                data_list = []
                y_object_list = []
                y_tool_list = []
                for tool in tools:
                    x, y = get_split_data_objects(binary_dataset_path, trials, objects_labels, args.robot, behavior,
                                                  modality, tool, objects, args.feature)
                    data_list.extend(x)
                    y_object_list.extend(y)
                    y_tool_list.extend(np.repeat(tools_labels[tool], len(x)))
                data_list = np.array(data_list)
                y_object_list = np.array(y_object_list)
                y_tool_list = np.array(y_tool_list)

                logging.info('data_list: {}'.format(data_list.shape))
                logging.info('y_object_list: {}, {}'.format(y_object_list.shape, y_object_list.flatten()[0:15]))
                logging.info('y_tool_list: {}, {}'.format(y_tool_list.shape, y_tool_list.flatten()[0:15]))

                X_reduced = dim_reduction_fn.fit_transform(data_list)
                logging.info('X_reduced: {}'.format(X_reduced.shape))

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
                fig.suptitle(args.robot + '-' + behavior.capitalize() + '-' + modality.capitalize(), fontsize='22')
                fig.subplots_adjust(top=0.92)  # title close to plot

                legend_elements1 = []
                for tool in tools_labels:
                    legend_elements1.append(
                        Line2D([], [], marker=tool_markers[tools_labels[tool]], color='w', label=tool,
                               markerfacecolor='k', markersize=12))
                legend_elements2 = []
                for obj_name in objects_labels:
                    obj_lab = objects_labels[obj_name]
                    legend_elements2.append(mpatches.Patch(color=object_colors[obj_lab], label=obj_name))

                for tool in tools_labels:
                    logging.info('tool: {}'.format(tool))
                    tool_lab = tools_labels[tool]
                    for obj_name in objects_labels:
                        logging.info('obj_name: {}'.format(obj_name))
                        obj_lab = objects_labels[obj_name]

                        indices = np.where((y_object_list == obj_lab) & (y_tool_list == tool_lab))
                        logging.info('indices: {}'.format(indices))

                        ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], c=object_colors[obj_lab], s=100,
                                   edgecolor='black', marker=tool_markers[tool_lab])

                legend1 = ax.legend(handles=legend_elements1, title='Tools:', fontsize=14, title_fontsize=15,
                                    bbox_to_anchor=bbox_to_anchor_tools, loc='upper right')
                legend2 = ax.legend(handles=legend_elements2, title='Objects:', fontsize=14, title_fontsize=15,
                                    bbox_to_anchor=bbox_to_anchor_objects, loc='upper right')
                ax.add_artist(legend1)
                ax.add_artist(legend2)
                ax.yaxis.set_tick_params(labelsize=14)
                ax.xaxis.set_tick_params(labelsize=14)

                plot_path = plot_dataset_path + os.sep + args.feature + os.sep
                os.makedirs(plot_path, exist_ok=True)
                plt.savefig(plot_path + '_'.join([args.robot, behavior, modality, args.dim_reduction]),
                            bbox_inches='tight', dpi=100)
                # plt.show(block=True)
                plt.close()
            else:
                # Plot multiple feature spaces in the same plot

                objects_labels = get_classes_labels(metadata[behaviors[0]]['objects'])
                logging.info('objects_labels: {}'.format(objects_labels))

                tools_labels = get_classes_labels(metadata[behaviors[0]]['tools'])
                logging.info('tools_labels: {}'.format(tools_labels))

                fig, axs_list = plt.subplots(nrows=1, ncols=len(args.feature), figsize=(5*len(args.feature), 5))
                fig.suptitle(args.robot + '-' + behavior.capitalize() + '-' + modality.capitalize(), fontsize='22')
                fig.subplots_adjust(top=0.85)  # title close to plot

                for plot_idx, feature in enumerate(args.feature):

                    path = binary_dataset_path if feature.startswith('discretized') else binary_dataset_path  # + '_AE'

                    data_list = []
                    y_object_list = []
                    y_tool_list = []
                    for tool in tools:
                        x, y = get_split_data_objects(path, trials, objects_labels, args.robot, behavior, modality,
                                                      tool, objects, feature)
                        data_list.extend(x)
                        y_object_list.extend(y)
                        y_tool_list.extend(np.repeat(tools_labels[tool], len(x)))
                    data_list = np.array(data_list)
                    y_object_list = np.array(y_object_list)
                    y_tool_list = np.array(y_tool_list).reshape((-1, 1))

                    logging.info('data_list: {}'.format(data_list.shape))
                    logging.info('y_object_list: {}, {}'.format(y_object_list.shape, y_object_list.flatten()[0:15]))
                    logging.info('y_tool_list: {}, {}'.format(y_tool_list.shape, y_tool_list.flatten()[0:15]))

                    X_reduced = dim_reduction_fn.fit_transform(data_list)
                    logging.info('X_reduced: {}'.format(X_reduced.shape))

                    for tool in tools_labels:
                        logging.info('tool: {}'.format(tool))
                        tool_lab = tools_labels[tool]
                        for obj_name in objects_labels:
                            logging.info('obj_name: {}'.format(obj_name))
                            obj_lab = objects_labels[obj_name]

                            indices = np.where((y_object_list == obj_lab) & (y_tool_list == tool_lab))
                            logging.info('indices: {}'.format(indices))

                            axs_list[plot_idx].scatter(X_reduced[indices, 0], X_reduced[indices, 1],
                                                       c=object_colors[obj_lab], s=100, edgecolor='black',
                                                       marker=tool_markers[tool_lab], alpha=0.8)

                    axs_list[plot_idx].set_title(f'{feature} Features')
                    axs_list[plot_idx].yaxis.set_tick_params(labelsize=14)
                    axs_list[plot_idx].xaxis.set_tick_params(labelsize=14)

                legend_elements1 = []
                for tool in tools_labels:
                    legend_elements1.append(
                        Line2D([], [], marker=tool_markers[tools_labels[tool]], color='w', label=tool,
                               markerfacecolor='k', markersize=12))

                legend_elements2 = []
                for obj_name in objects_labels:
                    obj_lab = objects_labels[obj_name]
                    legend_elements2.append(mpatches.Patch(color=object_colors[obj_lab], label=obj_name))

                legend1 = axs_list[plot_idx].legend(handles=legend_elements1, title='Tools:', fontsize=14,
                                                    title_fontsize=15, bbox_to_anchor=bbox_to_anchor_tools,
                                                    loc='upper right')
                legend2 = axs_list[plot_idx].legend(handles=legend_elements2, title='Objects:', fontsize=14,
                                                    title_fontsize=15, bbox_to_anchor=bbox_to_anchor_objects,
                                                    loc='upper right')
                axs_list[plot_idx].add_artist(legend1)
                axs_list[plot_idx].add_artist(legend2)

                plot_path = plot_dataset_path + os.sep + args.dim_reduction + os.sep
                os.makedirs(plot_path, exist_ok=True)
                plt.savefig(plot_path + '_'.join([args.robot, behavior, modality, args.dim_reduction]),
                            bbox_inches='tight', dpi=100)
                # plt.show(block=True)
                plt.close()

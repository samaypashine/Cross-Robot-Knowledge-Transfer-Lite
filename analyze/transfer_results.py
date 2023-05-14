# Author: Gyan Tatiya

import copy
import csv
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from paper5.utils import get_config, plot_all_modalities, fix_names, get_dim_reduction_fn, get_classes_labels


def write_all_results(path, results_dir, behaviors_to_skip=None, average_scores_n=0):

    if behaviors_to_skip is None:
        behaviors_to_skip = []

    all_modalities_type = 'all_modalities_train'
    with open(path + results_dir + os.sep + 'all_results.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Source Robot', 'Source Behavior', 'Source Tool', 'Target Robot','Target Behavior', 'Target Tool', 'Baseline 1 Accuracy',
                         'Baseline 1 SD', 'Transfer Accuracy', 'Transfer SD',
                         'Difference in Accuracy from Baseline 1'])

    projection_count = 0
    for projection in os.listdir(path):

        projection_list = projection.split('_TO_')
        if len(projection_list) <= 1:
            continue

        source_context, target_context = projection_list[0], projection_list[1]
        source_robot, source_behavior, source_tool = source_context.split('_')[0], source_context.split('_')[1], source_context.split('_')[2]
        target_robot, target_behavior, target_tool = target_context.split('_')[0], target_context.split('_')[1], target_context.split('_')[2]

        if (source_behavior in behaviors_to_skip) or (target_behavior in behaviors_to_skip):
            continue

        results = genfromtxt(path + os.sep + projection + os.sep + 'results_' + all_modalities_type + '.csv',
                             delimiter=',')[1:]

        if average_scores_n:
            bl1_accuracy, bl1_accuracy_sd = np.mean(results[average_scores_n:, 1]),\
                                            np.std(results[average_scores_n:, 1])
            # bl2_accuracy, bl2_accuracy_sd = np.mean(results[average_scores_n:, 3]),\
            #                                 np.std(results[average_scores_n:, 3])
            transfer_accuracy, transfer_accuracy_sd = np.mean(results[average_scores_n:, 5]),\
                                                      np.std(results[average_scores_n:, 5])
            difference_in_accuracy1 = np.mean(results[average_scores_n:, 7])
        else:
            results = results[-1]  # Take only last result

            bl1_accuracy, bl1_accuracy_sd = results[1], results[2]
            transfer_accuracy, transfer_accuracy_sd = results[3], results[4]
            difference_in_accuracy1 = results[5]

        with open(path + results_dir + os.sep + 'all_results.csv', 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([source_robot, source_behavior, source_tool, target_robot, target_behavior, target_tool, bl1_accuracy, bl1_accuracy_sd,
                             transfer_accuracy, transfer_accuracy_sd,
                             difference_in_accuracy1])
        projection_count += 1

    print('projection_count: ', projection_count)


def find_results_stats(path, results_dir):

    df = pd.read_csv(path + results_dir + os.sep + 'all_results.csv')
    print('\nNo. Projections: ', len(df))

    print('Average Baseline 1 Accuracy: ', np.mean(df['Baseline 1 Accuracy'].values.ravel()),
          np.std(df['Baseline 1 Accuracy'].values.ravel()))
    print('Average Transfer Accuracy: ', np.mean(df['Transfer Accuracy'].values.ravel()),
          np.std(df['Transfer Accuracy'].values.ravel()))

    df2 = df['Difference in Accuracy from Baseline 1']
    print('Count of Acc. Delta below 0 (Baseline 1): ', np.sum((df2 < 0).values.ravel()))
    print('Average Acc. Delta (Baseline 1): ', np.mean(df2.values.ravel()), np.std(df2.values.ravel()))

    print('')


def plot_best_worst(path, results_dir, n, best_or_worst, plot_bl2=False):

    df = pd.read_csv(path + results_dir + os.sep + 'all_results.csv')

    if best_or_worst == 'best':
        df2 = df.nsmallest(n, 'Difference in Accuracy from Baseline 1')
        title = 'Top ' + str(n) + ' Minimum Accuracy Delta Projections'
    elif best_or_worst == 'worst':
        df2 = df.nlargest(n, 'Difference in Accuracy from Baseline 1')
        title = 'Top ' + str(n) + ' Maximum Accuracy Delta Projections'
    # print(best_or_worst, df2)

    indices = np.arange(n)
    transfer_accuracy = np.array(df2['Transfer Accuracy']) * 100
    transfer_accuracy_sd = np.array(df2['Transfer SD']) * 100
    bl1_accuracy = np.array(df2['Baseline 1 Accuracy']) * 100
    bl1_accuracy_sd = np.array(df2['Baseline 1 SD']) * 100

    # difference_in_accuracy1 = np.array(df2['Difference in Accuracy from Baseline 1']) * 100
    # difference_in_accuracy2 = np.array(df2['Difference in Accuracy from Baseline 2']) * 100

    x_labels = []
    for projection in zip(list(df2['Source Robot']), list(df2['Source Behavior']), list(df2['Source Tool']), list(df2['Target Robot']), list(df2['Target Behavior']),
                          list(df2['Target Tool'])):
        projection = fix_names(projection)
        x_labels.append('-'.join([projection[0], projection[1], projection[2]]) + ' To\n' + '-'.join([projection[3], projection[4], projection[5]]))

    bar_width = 0.40
    if plot_bl2:
        bar_width = 0.25

    plt.subplots(figsize=(20, 10))

    plt.bar(indices, transfer_accuracy, width=bar_width, yerr=transfer_accuracy_sd, capsize=10, hatch='x',
            edgecolor='k', alpha=0.8, color='#89bc73', label='Transfer Condition (Trained on common latent features)')
    plt.bar(indices + bar_width, bl1_accuracy, width=bar_width, yerr=bl1_accuracy_sd, capsize=10, color='#ea52bf',
            hatch='o', edgecolor='k', alpha=0.8, label='Ground Truth Features (Trained on target context)')

    plt.yticks(fontsize=20)
    if plot_bl2:
        plt.xticks(indices + bar_width, x_labels, rotation=-15, fontsize=20)
    else:
        plt.xticks(indices + (bar_width / 2), x_labels, rotation=-15, fontsize=20)
    # plt.ylim(0, 100)
    plt.title(title, fontsize=20)
    plt.ylabel('% Recognition Accuracy', fontsize=20)

    if best_or_worst == 'best':
        plt.legend(fontsize=20)
    file_name = '_'.join(title.split(' '))
    plt.savefig(path + results_dir + os.sep + file_name + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()

    return df2


def plot_acc_curve(path, results_dir, df, clf_name, best_or_worst, plot_bl2=False):

    all_modalities_type = 'all_modalities_train'

    count = 1
    for projection in zip(list(df['Source Robot']), list(df['Source Behavior']), list(df['Source Tool']), list(df['Target Robot']), list(df['Target Behavior']),
                          list(df['Target Tool'])):
        source_robot, source_behavior, source_tool, target_robot, target_behavior, target_tool = projection[0], projection[1], projection[2], projection[3], projection[4], projection[5]
        projection_dir = '_'.join([source_robot, source_behavior, source_tool, 'TO', target_robot, target_behavior, target_tool, clf_name])
        results_file_path = path + os.sep + projection_dir + os.sep + 'results.bin'

        bin_file = open(results_file_path, 'rb')
        folds_proba_score_bl = pickle.load(bin_file)
        folds_proba_score_kt = pickle.load(bin_file)
        bin_file.close()

        source_robot, source_behavior, source_tool, target_robot, target_behavior, target_tool = fix_names([source_robot, source_behavior, source_tool,
                                                                                                            target_robot, target_behavior, target_tool])

        title_name = ' '.join([source_robot, source_behavior, source_tool, 'To', target_robot, target_behavior, target_tool])
        file_name = '_'.join([best_or_worst.capitalize(), str(count)] + title_name.split(' ') + [all_modalities_type])
        xlabel = 'Number of Shared Objects'
        plot_all_modalities(folds_proba_score_bl, folds_proba_score_kt, all_modalities_type,
                            title_name, xlabel, path + results_dir + os.sep, file_name, ylim=False, errorbar=True,
                            plot_bl2=plot_bl2)

        count += 1


def plot_acc_delta_matrix(path, results_dir, across, across_contexts, fixed_contexts_1, fixed_contexts_2):

    df = pd.read_csv(path + results_dir + os.sep + 'all_results.csv')

    acc_del_matrix_all_contexts = []
    for fixed_context_1 in sorted(fixed_contexts_1):
        for fixed_context_2 in sorted(fixed_contexts_2):
            print('fixed_context_1: ', fixed_context_1)
            print('fixed_context_2: ', fixed_context_2)

            acc_del_dict = {}
            across_loss = {context: 0.0 for context in across_contexts}

            for context in across_loss:
                acc_del_dict.setdefault(context, copy.deepcopy(across_loss))

            for s_across_context in across_contexts:
                s_context = fixed_context_1 + '_' + fixed_context_2 + '_' + s_across_context

                for t_across_context in across_contexts:
                    
                    if s_across_context != t_across_context:
                        t_context = fixed_context_1 + '_' + fixed_context_2 + '_' + t_across_context
                       
                        # if s_context == t_context:
                        #     # print('SAME Context')
                        #     continue

                        source_behavior, source_tool, source_robot = s_context.split('_')[0], s_context.split('_')[1], s_context.split('_')[2]
                        target_behavior, target_tool, target_robot = t_context.split('_')[0], t_context.split('_')[1], t_context.split('_')[2]

                        df_temp = df.loc[df['Source Robot'] == source_robot]
                        df_temp = df_temp.loc[df['Source Behavior'] == source_behavior]
                        df_temp = df_temp.loc[df['Source Tool'] == source_tool]
                        df_temp = df_temp.loc[df['Target Robot'] == target_robot]
                        df_temp = df_temp.loc[df['Target Behavior'] == target_behavior]
                        df_temp = df_temp.loc[df['Target Tool'] == target_tool]
                        
                        acc_del_dict[s_across_context][t_across_context] = float(df_temp['Difference in Accuracy from Baseline 1'])

        acc_del_matrix = []
        for s_across_context in sorted(acc_del_dict):
            temp = []
            for t_across_context in sorted(acc_del_dict[s_across_context]):
                temp.append(acc_del_dict[s_across_context][t_across_context])
            acc_del_matrix.append(temp)
        acc_del_matrix = np.array(acc_del_matrix)
        acc_del_matrix_all_contexts.append(acc_del_matrix)
        print('Mean Acc. Delta:', np.mean(acc_del_matrix), np.std(acc_del_matrix))

        plt.figure(figsize=(8, 8))
        plt.imshow(acc_del_matrix, cmap=plt.cm.gray, vmin=0, vmax=1)

        fixed_context_1 = fix_names([fixed_context_1])[0]
        fixed_context_2 = fix_names([fixed_context_2])[0]
        title = source_robot + '_' + fixed_context_1 + '_' + fixed_context_2 + ' to ' + target_robot + '_' + fixed_context_1 + '_' + fixed_context_2 + ' Projections'
        plt.title(title, fontsize=20)
        plt.xlabel('Target ' + across.capitalize(), fontsize=20)
        plt.ylabel('Source ' + across.capitalize(), fontsize=20)

        ax = plt.gca()

        tick_marks = np.arange(len(across_loss))
        ax.set_xticks(tick_marks, minor=False)
        xticklabels = fix_names(sorted(across_loss.keys()))
        ax.set_xticklabels(xticklabels, fontdict=None, minor=False, rotation=-15, fontsize=16)

        ax.set_yticks(tick_marks, minor=False)
        yticklabels = fix_names(sorted(across_loss.keys()))
        ax.set_yticklabels(yticklabels, fontdict=None, minor=False, fontsize=16)

        cb = plt.colorbar(fraction=0.046)

        font_size = 16
        cb.ax.tick_params(labelsize=font_size)

        file_name = 'Projections_' + source_robot + '_' + fixed_context_1 + '_' + fixed_context_2 + '_to_' + target_robot + '_' + fixed_context_1 + '_' + fixed_context_2
        plt.savefig(path + results_dir + os.sep + file_name + '.png', bbox_inches='tight', dpi=100)
        # plt.show()
        plt.close()

    acc_del_matrix_all_contexts = np.array(acc_del_matrix_all_contexts)
    acc_del_matrix_all_contexts = acc_del_matrix_all_contexts.reshape(len(across_contexts), -1)

    return acc_del_matrix_all_contexts


def plot_acc_delta_2d(acc_del_matrix, dim_reduction, across, across_context_list, path):

    dim_reduction_fn = get_dim_reduction_fn(dim_reduction)
    cxc_list_2d = dim_reduction_fn.fit_transform(acc_del_matrix)

    plt.figure(figsize=(7, 5))

    markers = ['o', '^', 's', 'P', '*', 'X', 'D']

    for i, across_context in enumerate(across_context_list):
        across_context = fix_names([across_context])[0]
        plt.scatter(cxc_list_2d[i][0], cxc_list_2d[i][1], edgecolor='k', s=100, marker=markers[i], label=across_context)

    plt.ylabel(dim_reduction + ' Feature 1')
    plt.xlabel(dim_reduction + ' Feature 2')
    plt.legend(fontsize=10, title=across.capitalize() + ':', title_fontsize=12)
    plt.savefig(path + os.sep + across.capitalize() + '_2D_neighborhood_graph.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_acc_delta_2d_v2(path, results_dir, across, tools, behaviors, dim_reduction):
    '''
    This is only for across tools and behaviors projections.
    It is similar to plot_acc_delta_matrix(), but it creates a accuracy delta matrix for evey possible projection and
    averages the accuracy delta from source to target and target and source.
    Then it plots it similar to plot_acc_delta_2d().
    '''

    df = pd.read_csv(path + results_dir + os.sep + 'all_results.csv')

    avg_acc_del_dict = {}
    for behavior in behaviors:
        for tool in tools:
            avg_acc_del_dict[behavior + '_' + tool] = 0.0

    cxc = {}
    for context in avg_acc_del_dict:
        cxc.setdefault(context, copy.deepcopy(avg_acc_del_dict))

    for s_context in sorted(cxc):
        for t_context in sorted(cxc[s_context]):

            if s_context == t_context:
                continue

            source_behavior, source_tool = s_context.split('_')[0], s_context.split('_')[1]
            target_behavior, target_tool = t_context.split('_')[0], t_context.split('_')[1]

            df_temp = df.loc[df['Source Behavior'] == source_behavior]
            df_temp = df_temp.loc[df['Source Tool'] == source_tool]
            df_temp = df_temp.loc[df['Target Behavior'] == target_behavior]
            df_temp = df_temp.loc[df['Target Tool'] == target_tool]
            acc_delta1 = float(df_temp['Difference in Accuracy from Baseline 1'])

            df_temp = df.loc[df['Source Behavior'] == target_behavior]
            df_temp = df_temp.loc[df['Source Tool'] == target_tool]
            df_temp = df_temp.loc[df['Target Behavior'] == source_behavior]
            df_temp = df_temp.loc[df['Target Tool'] == source_tool]
            acc_delta2 = float(df_temp['Difference in Accuracy from Baseline 1'])

            avg_acc_delta = (acc_delta1 + acc_delta2) / 2

            cxc[s_context][t_context] = avg_acc_delta

    cxc_df = pd.DataFrame(cxc)
    cxc_list = cxc_df.values

    dim_reduction_fn = get_dim_reduction_fn(dim_reduction)
    cxc_list_2d = dim_reduction_fn.fit_transform(cxc_list)

    colors = ['orangered', 'blue', 'darkgreen', 'orange', 'maroon', 'lightblue', 'magenta', 'olive', 'brown', 'cyan',
              'darkblue', 'beige', 'chartreuse', 'gold', 'green', 'grey', 'coral', 'black', 'khaki', 'orchid',
              'steelblue', 'chocolate', 'indigo', 'crimson', 'fuchsia']
    markers = ['o', '^', 's', 'P', '*', 'X', 'D']

    behaviors_labels = get_classes_labels(behaviors)
    tools_labels = get_classes_labels(tools)

    # plt.figure(figsize=(7, 5))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

    i = 0
    for index, row in cxc_df.iterrows():

        behavior, tool = index.split('_')[0], index.split('_')[1]

        plt.scatter(cxc_list_2d[i][0], cxc_list_2d[i][1], c=colors[behaviors_labels[behavior]], edgecolor='k', s=100,
                    marker=markers[tools_labels[tool]], label=index)
        i += 1

    legend_elements1 = []  # [Line2D([0], [0], color='w', label='Tools:')]
    for tool in tools:
        tool_ = fix_names([tool])[0]
        legend_elements1.append(Line2D([], [], marker=markers[tools_labels[tool]], color='w', label=tool_,
                                      markerfacecolor='k', markersize=12))

    legend_elements2 = []  # [Line2D([0], [0], color='w', label='Behaviors:')]
    for behavior in behaviors_labels:
        behavior_ = fix_names([behavior])[0]
        legend_elements2.append(mpatches.Patch(color=colors[behaviors_labels[behavior]], label=behavior_))

    # legend1 = ax.legend(handles=legend_elements1, fontsize=10, loc='upper left')
    # legend2 = ax.legend(handles=legend_elements2, fontsize=10, loc='upper right')
    legend1 = ax.legend(handles=legend_elements1, title='Tools:', fontsize=10, title_fontsize=12,
                        loc='upper left', framealpha=0.5)
    legend2 = ax.legend(handles=legend_elements2, title='Behaviors:', fontsize=10, title_fontsize=12,
                        loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # plt.ylabel(dim_reduction + ' Feature 1')
    # plt.xlabel(dim_reduction + ' Feature 2')
    # plt.legend(handles=legend_elements, fontsize=10, bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.savefig(path + os.sep + results_dir + os.sep + across.capitalize() + '_2D_neighborhood_graph.png',
                bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def get_behaviors_tools(path):

    df = pd.read_csv(path + os.sep + 'all_results.csv')

    robots = sorted(df['Source Robot'].unique())
    behaviors = sorted(df['Source Behavior'].unique())
    tools = sorted(df['Source Tool'].unique())

    return robots, behaviors, tools


if __name__ == '__main__':

    # results_path = r'/home/gyan/Documents/paper5/results/transfer_autoencoder-linear-tl_MLP/across_behaviors_aug_10_trials/'

    # results_path = r'/home/gyan/Documents/paper5/results/transfer_autoencoder-linear-tl_MLP/across_tools_aug_10_trials/'

    # results_path = r'/home/gyan/Documents/paper5/results/transfer_autoencoder-linear-tl/across_tools_behaviors_aug_10_trials/'
    
    results_path = r'/home/samaypashine/Desktop/Spring-2023/Directed/paper5/results/transfer_discretized-10-bins/across_robots_aug_2_trials'

    results_dir = 'result_plots'
    os.makedirs(results_path + results_dir, exist_ok=True)

    config = get_config(results_path + os.sep + 'config.yaml')
    print('config: ', config)

    across = config['across']
    print('across: ', across)

    classifier_name = config['classifier_name']
    print('classifier_name: ', classifier_name)

    behaviors_to_skip = ['1-look']  # ['1-look']
    average_scores_n = 0

    write_all_results(results_path, results_dir, behaviors_to_skip, average_scores_n)

    robots, behaviors, tools = get_behaviors_tools(results_path + results_dir)
    print('robots: ', robots)
    print('behaviors: ', behaviors)
    print('tools: ', tools)

    find_results_stats(results_path, results_dir)

    n = 5
    plot_bl2 = False
    data_frame = plot_best_worst(results_path, results_dir, n, 'best', plot_bl2=plot_bl2)
    plot_acc_curve(results_path, results_dir, data_frame, classifier_name, 'best', plot_bl2=plot_bl2)
    data_frame = plot_best_worst(results_path, results_dir, n, 'worst', plot_bl2=plot_bl2)
    plot_acc_curve(results_path, results_dir, data_frame, classifier_name, 'worst', plot_bl2=plot_bl2)

    dim_reduction = 'PCA'  # PCA, ISOMAP, TSNE
    if across == 'tools_behaviors':
        plot_acc_delta_2d_v2(results_path, results_dir, across, tools, behaviors, dim_reduction)
    elif across == 'robots':
        across_context_list = robots 
        fixed_context_list_1 = behaviors
        fixed_context_list_2 = tools
        
        acc_del_matrix_all_contexts = plot_acc_delta_matrix(results_path, results_dir, across, across_context_list, fixed_context_list_1, fixed_context_list_2)

        plot_acc_delta_2d(acc_del_matrix_all_contexts, dim_reduction, across, across_context_list, results_path + results_dir)
    else:
        across_context_list = tools
        fixed_context_list = behaviors
        if across == 'behaviors':
            across_context_list = behaviors
            fixed_context_list = tools
        acc_del_matrix_all_contexts = plot_acc_delta_matrix(results_path, results_dir, across, across_context_list, fixed_context_list)

        plot_acc_delta_2d(acc_del_matrix_all_contexts, dim_reduction, across, across_context_list, results_path + results_dir)

    '''
    TODO:
    
    '''

# Author: Gyan Tatiya

import argparse
import csv
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from paper5.utils import get_config, get_classes_labels, get_classifier, classifier, update_all_modalities, \
    update_all_behaviors_modalities, compute_mean_accuracy, save_config, get_split_data_objects, split_train_test_trials


if __name__ == '__main__':
    '''
    This script trains an object identity recognition model using a tool.
    Training and testing is done using 5-fold trial based cross validation.
    This is mainly to know the performance of each behavior and tool.

    Assumptions:
    When discretized features are used, for image modalities, only color histogram is used of last image in the video
    '''

    parser = argparse.ArgumentParser(description='Learn object recognition.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset', 'Tool_Dataset_Prob'],
                        # required=True,
                        default='Tool_Dataset_Prob',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5', 'gen3-lite'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-feature',
                        choices=['discretized-10-bins', 'discretized-20-bins', 'autoencoder-linear',
                                 'autoencoder-linear-tl'],
                        default='discretized-10-bins',
                        help='feature type')
    parser.add_argument('-vision-feature',
                        choices=['hist', 'resnet18'],
                        default='resnet18',
                        help='vision feature type')
    parser.add_argument('-classifier-name',
                        choices=['SVM-RBF', 'SVM-LIN', 'KNN', 'DT', 'RF', 'AB', 'GN', 'MLP'],
                        default='MLP',
                        help='classifier')
    parser.add_argument('-num-folds',
                        default=5,
                        type=int,
                        help='number of folds')
    args = parser.parse_args()

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    recognition_task = 'object'

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_path = 'results' + os.sep + 'classify_' + args.feature + os.sep + time_stamp + os.sep
    os.makedirs(results_path, exist_ok=True)

    log_path = results_path + time_stamp + '.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info("args: {}".format(args))

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    config.update(vars(args))
    logging.info('config: {}'.format(config))

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities = config[args.dataset][args.robot]['modalities']
    # modalities = [modality for modality in modalities if 'image' not in modality]  # Skipping image modalities
    logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([r'data', args.dataset + '_Binary', args.robot, 'dataset_metadata.bin'])
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

    objects_labels = get_classes_labels(metadata[behaviors[0]][recognition_task + 's'])
    logging.info('objects_labels: {}'.format(objects_labels))

    clf = get_classifier(args.classifier_name)

    num_of_test_examples = 1
    folds = len(trials) // num_of_test_examples
    folds_trials_split = split_train_test_trials(folds, len(trials))
    logging.info('folds_trials_split: {}'.format(folds_trials_split))

    for tool in tools:
        folds_behaviors_modalities_proba_score = {}
        for trial_fold in sorted(folds_trials_split):
            logging.info('trial_fold: {}'.format(trial_fold))
            folds_behaviors_modalities_proba_score.setdefault(trial_fold, {})
            for behavior in behaviors:
                folds_behaviors_modalities_proba_score[trial_fold].setdefault(behavior, {})
                # For each modality, combine weighted probability based on its accuracy score
                for modality in modalities:
                    logging.info('tool: {}'.format(tool))
                    logging.info('behavior: {}'.format(behavior))
                    logging.info('modality: {}'.format(modality))
                    folds_behaviors_modalities_proba_score[trial_fold][behavior].setdefault(modality, {})

                    # Get all objects to train the projection function
                    objects_list = list(objects_labels.keys())

                    # Get data
                    data, y = get_split_data_objects(binary_dataset_path, trials, objects_labels, args.robot, behavior,
                                                     modality, tool, objects_list, args.feature, args.vision_feature)
                    logging.info('data: {}'.format(data.shape))
                    logging.info('y: {}, {}'.format(y.shape, y.flatten()[0:15]))

                    # Reshaping data to access trials for training and testing
                    data2 = data.reshape(len(objects_list), -1, data.shape[-1])
                    y2 = y.reshape(len(objects_list), -1, y.shape[-1])
                    logging.info('data2: {}'.format(data2.shape))
                    logging.info('y2: {}, {}'.format(y2.shape, y.flatten()))

                    # Get train data
                    train_data2 = data2[:, folds_trials_split[trial_fold]['train']].reshape((-1, data2.shape[-1]))
                    train_y2 = y2[:, folds_trials_split[trial_fold]['train']].reshape((-1, 1))
                    logging.info('train_data2: {}'.format(train_data2.shape))
                    logging.info('train_y2: {}, {}'.format(train_y2.shape, train_y2.flatten()))

                    # Get test data
                    test_data2 = data2[:, folds_trials_split[trial_fold]['test']].reshape((-1, data2.shape[-1]))
                    test_y2 = y2[:, folds_trials_split[trial_fold]['test']].reshape((-1, 1))
                    logging.info('test_data2: {}'.format(test_data2.shape))
                    logging.info('test_y2: {}, {}'.format(test_y2.shape, test_y2.flatten()))

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(clf, train_data2, test_data2, train_y2, test_y2)
                    logging.info('y_prob_acc: {}, {}'.format(y_acc, y_pred))

                    folds_behaviors_modalities_proba_score[trial_fold][behavior][modality]['proba'] = y_proba
                    folds_behaviors_modalities_proba_score[trial_fold][behavior][modality]['test_acc'] = y_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(clf, train_data2, train_data2, train_y2, train_y2)
                    logging.info('y_prob_acc_train: {}, {}'.format(y_acc_train, y_pred_train))

                    folds_behaviors_modalities_proba_score[trial_fold][behavior][modality]['train_acc'] = y_acc_train

                folds_behaviors_modalities_proba_score[trial_fold][behavior] = \
                    update_all_modalities(folds_behaviors_modalities_proba_score[trial_fold][behavior], test_y2)

            folds_behaviors_modalities_proba_score[trial_fold] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score[trial_fold], test_y2)

        behaviors_modalities_score = compute_mean_accuracy(folds_behaviors_modalities_proba_score, vary_objects=False)

        for behavior in behaviors_modalities_score:
            logging.info('{}: {}'.format(behavior, behaviors_modalities_score[behavior]))

        row = ['behavior']
        for modality in modalities:
            logging.info('modality: {}'.format(modality))

            if modality in ['gripper_joint_states']:
                row.append('gripper')
            elif modality in ['camera_depth_image']:
                row.append('depth')
            elif modality in ['camera_rgb_image']:
                row.append('rgb')
            elif modality in ['touch_image']:
                row.append('touch')
            else:
                row.append(modality)

        df = pd.DataFrame(columns=row)
        for behavior in behaviors_modalities_score:

            if not behavior.startswith('all_behaviors_modalities'):
                row = {'behavior': behavior}
                for modality in behaviors_modalities_score[behavior]:
                    if modality in ['gripper_joint_states']:
                        row['gripper'] = round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2)
                    elif modality in ['camera_depth_image']:
                        row['depth'] = round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2)
                    elif modality in ['camera_rgb_image']:
                        row['rgb'] = round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2)
                    elif modality in ['touch_image']:
                        row['touch'] = round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2)
                    else:
                        row[modality] = round(behaviors_modalities_score[behavior][modality]['mean'] * 100, 2)
                df = pd.concat([df, pd.DataFrame.from_records([row])])

        logging.info('df:\n{}'.format(df))
        results_file_path = results_path + recognition_task + '_' + args.robot + '-' + tool \
                            + '_' + args.classifier_name + '_test-trial.csv'
        df.to_csv(results_file_path, index=False)

        with open(results_file_path, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")

            row = ['Average: ']
            for column in df:
                if column != 'behavior':
                    row.append(round(df[[column]].mean(axis=0)[column], 2))
            writer.writerow(row)

            writer.writerow(['all_behaviors_modalities: ',
                             round(behaviors_modalities_score['all_behaviors_modalities']['mean'] * 100, 2),
                             'all_behaviors_modalities_train: ',
                             round(behaviors_modalities_score['all_behaviors_modalities_train']['mean'] * 100, 2),
                             'all_behaviors_modalities_test: ',
                             round(behaviors_modalities_score['all_behaviors_modalities_test']['mean'] * 100, 2)])

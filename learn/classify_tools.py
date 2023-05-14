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
    update_all_behaviors_modalities, compute_mean_accuracy, save_config, get_split_data_tools


if __name__ == '__main__':
    '''
    This script trains a tool identity recognition model using an object and test on all the objects, computes drop in
    performance when the test object is different than the train object.
    This is mainly to know the performance drop when the object changes.

    Assumptions:
    When discretized features are used, for image modalities, only color histogram is used of last image in the video
    '''


    parser = argparse.ArgumentParser(description='Learn tool recognition.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset'],
                        # required=True,
                        default='Tool_Dataset',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-train-object',
                        choices=['chickpea', 'wheat', 'cane-sugar', 'styrofoam-bead', 'water', 'metal-nut-bolt', 'salt',
                                 'split-green-pea', 'chickpea', 'detergent', 'plastic-bead', 'empty', 'chia-seed',
                                 'wooden-button', 'glass-bead', 'kidney-bean', 'wheat'],
                        default='cane-sugar',
                        help='object name')
    parser.add_argument('-feature',
                        choices=['discretized-10-bins', 'discretized-20-bins', 'autoencoder-linear',
                                 'autoencoder-linear-tl'],
                        default='discretized-10-bins',
                        help='feature type')
    parser.add_argument('-classifier-name',
                        choices=['SVM-RBF', 'SVM-LIN', 'KNN', 'DT', 'RF', 'AB', 'GN', 'MLP'],
                        default='KNN',
                        help='classifier')
    parser.add_argument('-num-folds',
                        default=5,
                        type=int,
                        help='number of folds')
    args = parser.parse_args()

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    recognition_task = 'tool'

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    results_path = 'results' + os.sep + f'classify_{args.feature}' + os.sep + time_stamp + os.sep
    os.makedirs(results_path, exist_ok=True)

    log_path = results_path + time_stamp + '.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

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

    classes_labels = get_classes_labels(metadata[behaviors[0]][recognition_task + 's'])
    logging.info('classes_labels: {}'.format(classes_labels))

    clf = get_classifier(args.classifier_name)

    folds = ['fold_' + str(fold) for fold in range(args.num_folds)]
    logging.info('folds: {}'.format(folds))

    for test_object in objects:
        logging.info('test_object: {}'.format(test_object))
        folds_behaviors_modalities_proba_score = {}
        for fold in sorted(folds):
            logging.info('fold: {}'.format(fold))
            folds_behaviors_modalities_proba_score.setdefault(fold, {})
            for behavior in behaviors:
                logging.info('behavior: {}'.format(behavior))
                folds_behaviors_modalities_proba_score[fold].setdefault(behavior, {})
                # For each modality, combine weighted probability based on its accuracy score
                for modality in modalities:
                    logging.info('modality: {}'.format(modality))
                    folds_behaviors_modalities_proba_score[fold][behavior].setdefault(modality, {})

                    X_train, y_train = get_split_data_tools(binary_dataset_path, trials, classes_labels, args.robot,
                                                            behavior, modality, tools, args.train_object, args.feature)
                    X_test, y_test = get_split_data_tools(binary_dataset_path, trials, classes_labels, args.robot,
                                                          behavior, modality, tools, test_object, args.feature)

                    logging.info('X_train: {}'.format(X_train.shape))
                    logging.info('y_train: {}, {}'.format(y_train.shape, y_train.flatten()))
                    logging.info('X_test: {}'.format(X_test.shape))
                    logging.info('y_test: {}, {}'.format(y_test.shape, y_test.flatten()))

                    # Train and Test
                    y_acc, y_pred, y_proba = classifier(clf, X_train, X_test, y_train, y_test)
                    logging.info('y_prob_acc: {}, {}'.format(y_acc, y_pred))

                    folds_behaviors_modalities_proba_score[fold][behavior][modality]['proba'] = y_proba
                    folds_behaviors_modalities_proba_score[fold][behavior][modality]['test_acc'] = y_acc

                    # For each behavior, get an accuracy score to combine weighted probability based on its accuracy score
                    # Use only training data to get a score
                    y_acc_train, y_pred_train, y_proba_train = classifier(clf, X_train, X_train, y_train, y_train)
                    logging.info('y_prob_acc_train: {}, {}'.format(y_acc_train, y_pred_train))

                    folds_behaviors_modalities_proba_score[fold][behavior][modality]['train_acc'] = y_acc_train

                folds_behaviors_modalities_proba_score[fold][behavior] = \
                    update_all_modalities(folds_behaviors_modalities_proba_score[fold][behavior], y_test)

            folds_behaviors_modalities_proba_score[fold] = \
                update_all_behaviors_modalities(folds_behaviors_modalities_proba_score[fold], y_test)

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
        results_file_path = results_path + recognition_task + '_' + args.robot + '-' + args.train_object \
                            + '_' + 'test-object-' + test_object + '_' + args.classifier_name + '_test-trial.csv'
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

    results_df = {}
    for test_object in objects:
        results_file_path = results_path + recognition_task + '_' + args.robot + '-' + args.train_object \
                            + '_' + 'test-object-' + test_object + '_' + args.classifier_name + '_test-trial.csv'
        df = pd.read_csv(results_file_path, delimiter=',')
        results_df[test_object] = df

    columns = results_df[args.train_object].columns
    columns_all_behaviors_modalities = list(results_df[args.train_object].columns)[1:6:2]
    columns_all_behaviors_modalities2 = list(results_df[args.train_object].columns)[2:6:2]
    columns_all_behaviors_modalities3 = ['all_behaviors_modalities_train', 'all_behaviors_modalities_test']

    for test_object in objects:
        if test_object == args.train_object:
            continue

        df = pd.DataFrame(columns=list(results_df[args.train_object].columns))
        for index, row in results_df[args.train_object].iterrows():

            row2 = {'behavior': row['behavior']}
            if index == len(results_df[args.train_object]) - 1:
                for c in columns_all_behaviors_modalities:
                    drop = float(results_df[test_object].loc[index][c]) - float(results_df[args.train_object].loc[index][c])
                    row2[c] = drop
                for i, c in enumerate(columns_all_behaviors_modalities2):
                    row2[c] = columns_all_behaviors_modalities3[i]
                df = pd.concat([df, pd.DataFrame.from_records([row2])])
                continue

            for c in columns:
                if c != 'behavior':
                    drop = float(results_df[test_object].loc[index][c]) - float(results_df[args.train_object].loc[index][c])
                    row2[c] = drop

            df = pd.concat([df, pd.DataFrame.from_records([row2])])

        results_file_path = results_path + recognition_task + '_' + args.robot + '-' + args.train_object \
                            + '_' + 'test-object-' + test_object + '_' + args.classifier_name + '_test-trial_drop.csv'
        df.to_csv(results_file_path, index=False)

    drops = []
    for test_object in objects:
        if test_object == args.train_object:
            continue
        results_file_path = results_path + recognition_task + '_' + args.robot + '-' + args.train_object \
                            + '_' + 'test-object-' + test_object + '_' + args.classifier_name + '_test-trial_drop.csv'
        df = pd.read_csv(results_file_path, delimiter=',')
        drops.append(df.loc[len(df) - 1][columns[3]])  # 3 for all_behaviors_modalities_train

    logging.info('Average drop across all objects {}: {}'.format(drops, np.mean(drops)))

    save_config(config, results_path + 'config.yaml')

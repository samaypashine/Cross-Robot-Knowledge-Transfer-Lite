# Author: Gyan Tatiya

import logging
import os
import pickle

import cv2
import librosa
import numpy as np
import yaml

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import torch
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms

from paper5.pytorch.beats.BEATs import BEATs, BEATsConfig


def time_taken(start, end):
    '''Human readable time between `start` and `end`
    :param start: time.time()
    :param end: time.time()
    :returns: day:hour:minute:second.millisecond
    '''

    my_time = end - start
    day = my_time // (24 * 3600)
    my_time = my_time % (24 * 3600)
    hour = my_time // 3600
    my_time %= 3600
    minutes = my_time // 60
    my_time %= 60
    seconds = my_time
    milliseconds = ((end - start) - int(end - start))
    day_hour_min_sec = str('%02d' % int(day)) + ':' + str('%02d' % int(hour)) + ':' + str('%02d' % int(minutes)) + \
                       ':' + str('%02d' % int(seconds) + '.' + str('%.3f' % milliseconds)[2:])

    return day_hour_min_sec


def fix_names(names):

    names = list(names)
    for i, name in enumerate(names):
        if name in ['1-look', '2-stirring-slow', '3-stirring-fast', '4-stirring-twist', '5-whisk', '6-poke']:
            names[i] = '-'.join([x.capitalize() for x in name[2:].split('-')])
        elif name in ['plastic-knife', 'metal-whisk', 'wooden-chopstick', 'plastic-spoon', 'metal-scissor',
                      'wooden-fork']:
            names[i] = '-'.join([x.capitalize() for x in name.split('-')])
        elif name in ['camera_depth_image', 'camera_rgb_image', 'touch_image', 'audio', 'gripper_joint_states',
                      'effort', 'position', 'velocity', 'torque', 'force']:
            if 'depth' in name:
                names[i] = 'Depth-Image'
            elif 'rgb' in name:
                names[i] = 'RGB-Image'
            elif 'gripper' in name:
                names[i] = 'Gripper'
            else:
                names[i] = name.capitalize()

    return names


def get_config(config_path):
    with open(config_path) as file:
        return yaml.safe_load(file)


def save_config(config, config_filepath):
    with open(config_filepath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def read_images(root_, files_):
    data_ = []
    for filename_ in sorted(files_):
        image = cv2.imread(root_ + os.sep + filename_, cv2.IMREAD_UNCHANGED)
        data_.append(image)

    return np.array(data_)


def add_nose_time_shift(data):
    # Shifting time dimension noise

    roll_by = np.random.uniform(0.01, 0.05)
    roll_by = int(roll_by * len(data))

    return np.roll(data, roll_by, axis=0)  # Rolling time dimension slightly


def add_noise_salt_pepper(data):
    # Salt and pepper noise

    num_features = np.prod(data.shape)
    shape = data.shape
    s_vs_p = 0.5
    amount = 0.004

    # Salt mode
    num_salt = np.ceil(amount * num_features * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) if i > 1 else np.repeat(0, int(num_salt)) for i in shape]
    for i in zip(*coords):
        data[i] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * num_features * (1 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) if i > 1 else np.repeat(0, int(num_pepper)) for i in shape]
    for i in zip(*coords):
        data[i] = 0

    return data


def get_classes_labels(objects_list):

    classes_labels_ = {}
    for i, object_name in enumerate(sorted(objects_list)):
        classes_labels_[object_name] = i

    return classes_labels_


def get_new_labels(y_object, objects_labels):

    label_count = 0
    y_object_new = []
    old_labels_new_label = {}
    objects_labels_new = {}
    for old_label in y_object.flatten():
        if old_label not in old_labels_new_label:
            old_labels_new_label[old_label] = label_count
            y_object_new.append(label_count)
            object_name_ = list(objects_labels.keys())[list(objects_labels.values()).index(old_label)]
            objects_labels_new[object_name_] = label_count
            label_count += 1
        else:
            y_object_new.append(old_labels_new_label[old_label])
    y_object_new = np.array(y_object_new).reshape((-1, 1))

    return y_object_new, objects_labels_new, old_labels_new_label


def split_train_test_trials(n_folds, trials_per_class):

    test_size = trials_per_class // n_folds
    tt_splits = {}

    for a_fold in range(n_folds):

        train_index = []
        test_index = np.arange(test_size * a_fold, test_size * (a_fold + 1))

        if test_size * a_fold > 0:
            train_index.extend(np.arange(0, test_size * a_fold))
        if test_size * (a_fold + 1) - 1 < trials_per_class - 1:
            train_index.extend(np.arange(test_size * (a_fold + 1), trials_per_class))

        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('train', []).extend(train_index)
        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('test', []).extend(test_index)

    return tt_splits


def split_train_test_objects(n_folds, objects, test_percentage):

    num_of_test_objects = int(len(objects) * test_percentage)
    tt_splits = {}
    for a_fold in range(n_folds):
        test_objects = np.random.choice(objects, size=num_of_test_objects, replace=False).tolist()
        train_objects = list(set(objects) - set(test_objects))

        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('test', test_objects)
        tt_splits.setdefault('fold_' + str(a_fold), {}).setdefault('train', train_objects)

    return tt_splits


def get_split_data_objects(path, trials, classes_labels, robot, behavior, modality, tool, objects, feature,
                           vision_feature=None):

    if isinstance(list(trials)[0], np.integer):
        trials = ['trial-' + str(trial_num) for trial_num in sorted(trials)]

    x_split = []
    y_split = []

    if 'resnet18' == vision_feature:
        # Initialize model with the best available weights
        weights = models.ResNet18_Weights.DEFAULT
        vision_model = models.resnet18(weights=weights)
        vision_model.eval()
        return_nodes = ['flatten']
        feature_extractor = create_feature_extractor(vision_model, return_nodes=return_nodes)

        transform_to_PIL = transforms.ToPILImage()
        preprocess = weights.transforms()  # Initialize the inference transforms

    if 'audio' in modality:
        checkpoint = torch.load('/cluster/tufts/sinapovlab/spashi01/workspace/paper5/paper5/pytorch/beats/BEATs_iter3_plus_AS2M.pt')
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()

    for object_name in sorted(objects):
        for trial_num in sorted(trials):
            if feature.startswith('discretized') and 'audio' in modality:
                trial_list = os.listdir(os.sep.join(["_".join(path.split('_')[:-1]), robot + "_" + tool, object_name]))
                TRIAL = [i for i in trial_list if trial_num in i][0]
                audio_path = os.sep.join(
                    ["_".join(path.split('_')[:-1]), robot + "_" + tool, object_name, TRIAL, behavior, modality, "audio.wav"])
                # 1. create time series of 10 seconds audio data
                audio_time_series, _ = librosa.load(audio_path, sr=16_000, duration=10)
                audio_time_series = audio_time_series[:160000]
                # 2. Extract features
                audio_time_series = torch.unsqueeze(torch.tensor(audio_time_series), 0)
                data = BEATs_model.extract_features(audio_time_series, padding_mask=None)[0]
                logging.info(f"😇😇{object_name}-{tool}-{behavior}-{trial_num} audio feature shape: {data.shape} ")
                data = data.detach().numpy()[:200, :150]  # don't need all the features

            if feature.startswith('discretized') and 'image' in modality:
                trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                   '-last-image.bin'])
                last_image_exists = True
                if not os.path.exists(trial_data_filepath):
                    last_image_exists = False
                    trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                       '.bin'])
                    if not os.path.exists(trial_data_filepath):
                        raw_dataset_path = '_'.join(path.split('_')[:-1])
                        trials_timestamp = os.listdir(os.sep.join([raw_dataset_path, robot + '_' + tool, object_name]))
                        for trial_timestamp in trials_timestamp:
                            trial = trial_timestamp.split('_')[0]
                            if trial == trial_num:
                                trial_num_timestamp = trial_timestamp
                                break

                        trial_data_path = os.sep.join([raw_dataset_path, robot + '_' + tool, object_name,
                                                       trial_num_timestamp, behavior, modality])
                        files = sorted(os.listdir(trial_data_path))
                        data = read_images(trial_data_path, [files[-1]])[0]  # Use last image

                        trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num,
                                                           modality + '-last-image.bin'])
                        output_file = open(trial_data_filepath, 'wb')
                        pickle.dump(data, output_file)
                        output_file.close()
                        last_image_exists = True
            else:
                trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                   '-' + feature + '.bin'])
            bin_file = open(trial_data_filepath, 'rb')
            data = pickle.load(bin_file)
            bin_file.close()

            if feature in ['autoencoder-linear', 'autoencoder-linear-tl']:
                data = data['code']

            if feature.startswith('discretized') and 'image' in modality:
                last_image = data  # Use last image
                if not last_image_exists:
                    last_image = data[-1]
                    trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                       '-last-image.bin'])
                    output_file = open(trial_data_filepath, 'wb')
                    pickle.dump(last_image, output_file)
                    output_file.close()

                bins = 4
                if 'depth' in modality:
                    if 'resnet18' == vision_feature:
                        last_image = last_image[:, :, np.newaxis]
                        last_image = np.repeat(last_image, 3, axis=-1)
                        last_image = transform_to_PIL(last_image)
                        batch = preprocess(last_image).unsqueeze(0)  # Apply inference preprocessing transforms
                        with torch.no_grad():
                            data = feature_extractor(batch)[return_nodes[0]].numpy()
                    elif 'hist' == vision_feature:
                        data = cv2.calcHist([last_image], [0], None, [bins], [0, 256])

                else:
                    if 'resnet18' == vision_feature:
                        last_image = transform_to_PIL(last_image)
                        batch = preprocess(last_image).unsqueeze(0)  # Apply inference preprocessing transforms
                        with torch.no_grad():
                            data = feature_extractor(batch)[return_nodes[0]].numpy()
                    elif 'hist' in vision_feature:
                        data = cv2.calcHist([last_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
                # logging.info('3D histogram shape: {}, with {} values'.format(data.shape, data.flatten().shape[0]))
            x_split.append(data.flatten())
            y_split.append(classes_labels[object_name])

    return np.array(x_split), np.array(y_split).reshape((-1, 1))


def get_split_data_tools(path, trials, classes_labels, robot, behavior, modality, tools, object_name, feature):
    if isinstance(list(trials)[0], np.integer):
        trials = ["trial-" + str(trial_num) for trial_num in sorted(trials)]

    x_split = []
    y_split = []
    for tool in sorted(tools):
        for trial_num in sorted(trials):
            if feature.startswith('discretized') and 'image' in modality:
                trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                   '-last-image.bin'])
                last_image_exists = True
                if not os.path.exists(trial_data_filepath):
                    last_image_exists = False
                    trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                       '.bin'])
            else:
                trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                   '-' + feature + '.bin'])
            bin_file = open(trial_data_filepath, 'rb')
            data = pickle.load(bin_file)
            bin_file.close()

            if feature in ['autoencoder-linear', 'autoencoder-linear-tl']:
                data = data['code']

            if feature.startswith('discretized') and 'image' in modality:
                last_image = data  # Use last image
                if not last_image_exists:
                    last_image = data[-1]
                    trial_data_filepath = os.sep.join([path, robot, behavior, object_name, tool, trial_num, modality +
                                                       '-last-image.bin'])
                    output_file = open(trial_data_filepath, 'wb')
                    pickle.dump(last_image, output_file)
                    output_file.close()

                bins = 4
                if 'depth' in modality:
                    data = cv2.calcHist([last_image], [0], None, [bins], [0, 256])
                else:
                    data = cv2.calcHist([last_image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
                # logging.info('3D histogram shape: {}, with {} values'.format(data.shape, data.flatten().shape[0]))
            x_split.append(data.flatten())
            y_split.append(classes_labels[tool])

    return np.array(x_split), np.array(y_split).reshape((-1, 1))


def augment_trials(X_data, y_object, num_trials_aug=5, object_labels=None, shuffle=True):
    # If object_labels is give then only augment these labels

    if object_labels is None:
        object_labels = set(y_object.flatten())

    X_data_aug = []
    y_object_aug = []
    for label in object_labels:
        indices = np.where(y_object == label)

        X_data_mean = np.mean(X_data[indices[0]], axis=0)
        X_data_std = np.std(X_data[indices[0]], axis=0)

        for _ in range(num_trials_aug):
            data_point = np.random.normal(X_data_mean, X_data_std)
            X_data_aug.append(data_point)
            y_object_aug.append(label)

    X_data_aug = np.array(X_data_aug)
    y_object_aug = np.array(y_object_aug).reshape((-1, 1))

    if len(X_data_aug) > 0:
        X_data = np.concatenate((X_data, X_data_aug), axis=0)
        y_object = np.concatenate((y_object, y_object_aug), axis=0)

    if shuffle:
        random_idx = np.random.permutation(X_data.shape[0])
        X_data = X_data[random_idx]
        y_object = y_object[random_idx]

    return X_data, y_object


def check_kema_data(kema_data):
    for x_key in kema_data:
        if 'Test' not in x_key and x_key.startswith('X') and kema_data[x_key].shape[0] <= 10:
            y_key = 'Y' + x_key[1]
            print('<= 10 EXAMPLES FOR: ', x_key, y_key)

            while kema_data[x_key].shape[0] <= 10:
                idx = np.random.choice(kema_data[x_key].shape[0])
                kema_data[x_key] = np.append(kema_data[x_key], kema_data[x_key][idx].reshape(1, -1), axis=0)
                kema_data[y_key] = np.append(kema_data[y_key], kema_data[y_key][idx].reshape(1, -1), axis=0)

    return kema_data


def get_classifier(name):
    if name == 'SVM-RBF':
        clf = SVC(gamma='auto', kernel='rbf', probability=True)
    elif name == 'SVM-LIN':
        clf = SVC(gamma='auto', kernel='linear', probability=True)
    elif name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=1)
    elif name == 'DT':
        clf = DecisionTreeClassifier()
    elif name == 'RF':
        clf = RandomForestClassifier()
    elif name == 'AB':
        clf = AdaBoostClassifier()
    elif name == 'GN':
        clf = GaussianNB()
    elif name == 'MLP':
        clf = MLPClassifier(random_state=0, max_iter=1000)
    else:
        raise Exception(name + ' does not exits!')

    return clf


def get_dim_reduction_fn(name):
    if name == 'ISOMAP':
        dim_reduction_fn = Isomap(n_neighbors=5, n_components=2)
    elif name == 'PCA':
        dim_reduction_fn = PCA(n_components=2)
    elif name == 'TSNE':
        dim_reduction_fn = TSNE(n_components=2, random_state=0)
    else:
        raise Exception(name + ' does not exits!')

    return dim_reduction_fn


def classifier(my_classifier, x_train, x_test, y_train, y_test):
    # Train a classifier on test data and return accuracy and prediction on test data

    # Fit the model on the training data
    my_classifier.fit(x_train, y_train.ravel())

    # See how the model performs on the test data
    # accuracy = my_classifier.score(x_test, y_test)
    # prediction = my_classifier.predict(x_test)
    probability = my_classifier.predict_proba(x_test)

    prediction = np.argmax(probability, axis=1)
    accuracy = np.mean(y_test.ravel() == prediction)

    return accuracy, prediction, probability


def combine_probability(proba_acc_list_, y_test_, acc=None):
    # For each classifier, combine weighted probability based on its accuracy score
    proba_list = []
    for proba_acc in proba_acc_list_:
        y_proba = proba_acc['proba']
        if acc and proba_acc[acc] > 0:
            # Multiply the score by probability to combine each classifier's performance accordingly
            # IMPORTANT: This will discard probability when the accuracy is 0
            y_proba = y_proba * proba_acc[acc]  # weighted probability
            proba_list.append(y_proba)
        elif not acc:
            proba_list.append(y_proba)  # Uniform combination, probability is combined even when the accuracy is 0

    # If all the accuracy is 0 in proba_acc_list_, the fill proba_list with chance accuracy
    if len(proba_list) == 0:
        num_examples, num_classes = proba_acc_list_[0]['proba'].shape
        chance_prob = (100 / num_classes) / 100
        proba_list = np.full((1, num_examples, num_classes), chance_prob)

    # Combine weighted probability of all classifiers
    y_proba_norm = np.zeros(len(proba_list[0][0]))
    for proba in proba_list:
        y_proba_norm = y_proba_norm + proba

    # Normalizing probability
    y_proba_norm_sum = np.sum(y_proba_norm, axis=1)  # sum of weighted probability
    y_proba_norm_sum = np.repeat(y_proba_norm_sum, len(proba_list[0][0]), axis=0).reshape(y_proba_norm.shape)
    y_proba_norm = y_proba_norm / y_proba_norm_sum

    y_proba_pred = np.argmax(y_proba_norm, axis=1)
    y_prob_acc = np.mean(y_test_ == y_proba_pred)

    return y_proba_norm, y_prob_acc


def update_all_modalities(modalities_proba_score, y_test_):
    # For each modality, combine weighted probability based on its accuracy score

    proba_acc_list = []
    for modality_ in modalities_proba_score:
        proba_acc = {'proba': modalities_proba_score[modality_]['proba'],
                     'train_acc': modalities_proba_score[modality_]['train_acc'],
                     'test_acc': modalities_proba_score[modality_]['test_acc']}
        proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    modalities_proba_score.setdefault('all_modalities', {})
    modalities_proba_score['all_modalities']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    modalities_proba_score.setdefault('all_modalities_train', {})
    modalities_proba_score['all_modalities_train']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    modalities_proba_score.setdefault('all_modalities_test', {})
    modalities_proba_score['all_modalities_test']['proba'] = y_proba_norm
    modalities_proba_score['all_modalities_test']['test_acc'] = y_prob_acc

    return modalities_proba_score


def update_all_behaviors_modalities(behaviors_modalities_proba_score, y_test_):
    # For each behavior and modality, combine weighted probability based on its accuracy score

    proba_acc_list = []
    for behavior_ in behaviors_modalities_proba_score:
        for modality_ in behaviors_modalities_proba_score[behavior_]:
            if not modality_.startswith('all_modalities'):
                proba_acc = {'proba': behaviors_modalities_proba_score[behavior_][modality_]['proba'],
                             'train_acc': behaviors_modalities_proba_score[behavior_][modality_]['train_acc'],
                             'test_acc': behaviors_modalities_proba_score[behavior_][modality_]['test_acc']}
                proba_acc_list.append(proba_acc)

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel())
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities', {})
    behaviors_modalities_proba_score['all_behaviors_modalities']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'train_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_train', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_train']['test_acc'] = y_prob_acc

    y_proba_norm, y_prob_acc = combine_probability(proba_acc_list, y_test_.ravel(), 'test_acc')
    behaviors_modalities_proba_score.setdefault('all_behaviors_modalities_test', {})
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['proba'] = y_proba_norm
    behaviors_modalities_proba_score['all_behaviors_modalities_test']['test_acc'] = y_prob_acc

    return behaviors_modalities_proba_score


def compute_mean_accuracy(folds_behaviors_modalities_proba_score, acc='test_acc', vary_objects=True,
                          behavior_present=True):
    behaviors_modalities_score = {}
    for fold_ in folds_behaviors_modalities_proba_score:
        if vary_objects:
            for objects_per_label_ in folds_behaviors_modalities_proba_score[fold_]:
                behaviors_modalities_score.setdefault(objects_per_label_, {})
                if behavior_present:
                    for behavior_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        if behavior_.startswith('all_behaviors_modalities'):
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][
                                acc]
                            behaviors_modalities_score[objects_per_label_][behavior_].append(y_prob_acc)
                        else:
                            behaviors_modalities_score[objects_per_label_].setdefault(behavior_, {})
                            for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_][
                                behavior_]:
                                behaviors_modalities_score[objects_per_label_][behavior_].setdefault(modality_, [])
                                y_prob_acc = \
                                    folds_behaviors_modalities_proba_score[fold_][objects_per_label_][behavior_][
                                        modality_][
                                        acc]
                                behaviors_modalities_score[objects_per_label_][behavior_][modality_].append(y_prob_acc)
                else:
                    for modality_ in folds_behaviors_modalities_proba_score[fold_][objects_per_label_]:
                        behaviors_modalities_score[objects_per_label_].setdefault(modality_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][objects_per_label_][modality_][acc]
                        behaviors_modalities_score[objects_per_label_][modality_].append(y_prob_acc)
        else:
            if behavior_present:
                for behavior_ in folds_behaviors_modalities_proba_score[fold_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score.setdefault(behavior_, [])
                        y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][acc]
                        behaviors_modalities_score[behavior_].append(y_prob_acc)
                    else:
                        behaviors_modalities_score.setdefault(behavior_, {})
                        for modality_ in folds_behaviors_modalities_proba_score[fold_][behavior_]:
                            behaviors_modalities_score[behavior_].setdefault(modality_, [])
                            y_prob_acc = folds_behaviors_modalities_proba_score[fold_][behavior_][modality_][acc]
                            behaviors_modalities_score[behavior_][modality_].append(y_prob_acc)
            else:
                for modality_ in folds_behaviors_modalities_proba_score[fold_]:
                    behaviors_modalities_score.setdefault(modality_, [])
                    y_prob_acc = folds_behaviors_modalities_proba_score[fold_][modality_][acc]
                    behaviors_modalities_score[modality_].append(y_prob_acc)

    if vary_objects:
        for objects_per_label_ in behaviors_modalities_score:
            if behavior_present:
                for behavior_ in behaviors_modalities_score[objects_per_label_]:
                    if behavior_.startswith('all_behaviors_modalities'):
                        behaviors_modalities_score[objects_per_label_][behavior_] = {
                            'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_]),
                            'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_])}
                    else:
                        for modality_ in behaviors_modalities_score[objects_per_label_][behavior_]:
                            behaviors_modalities_score[objects_per_label_][behavior_][modality_] = {
                                'mean': np.mean(behaviors_modalities_score[objects_per_label_][behavior_][modality_]),
                                'std': np.std(behaviors_modalities_score[objects_per_label_][behavior_][modality_])}
            else:
                for modality_ in behaviors_modalities_score[objects_per_label_]:
                    behaviors_modalities_score[objects_per_label_][modality_] = {
                        'mean': np.mean(behaviors_modalities_score[objects_per_label_][modality_]),
                        'std': np.std(behaviors_modalities_score[objects_per_label_][modality_])}
    else:
        if behavior_present:
            for behavior_ in behaviors_modalities_score:
                if behavior_.startswith('all_behaviors_modalities'):
                    behaviors_modalities_score[behavior_] = {
                        'mean': np.mean(behaviors_modalities_score[behavior_]),
                        'std': np.std(behaviors_modalities_score[behavior_])}
                else:
                    for modality_ in behaviors_modalities_score[behavior_]:
                        behaviors_modalities_score[behavior_][modality_] = {
                            'mean': np.mean(behaviors_modalities_score[behavior_][modality_]),
                            'std': np.std(behaviors_modalities_score[behavior_][modality_])}
        else:
            for modality_ in behaviors_modalities_score:
                behaviors_modalities_score[modality_] = {
                    'mean': np.mean(behaviors_modalities_score[modality_]),
                    'std': np.std(behaviors_modalities_score[modality_])}

    return behaviors_modalities_score


def plot_fold_all_modalities_v2(folds_proba_score_bl, folds_proba_score_kt, all_modalities_type,
                             title_name, xlabel, file_path, ylim=True, xticks=True):
    acc_bl = []
    acc_kt = []
    x_points = []
    for num_obj in folds_proba_score_bl:
        x_points.append(num_obj)
        acc_bl.append(folds_proba_score_bl[num_obj][all_modalities_type]['test_acc'])
        acc_kt.append(folds_proba_score_kt[num_obj][all_modalities_type]['test_acc'])
    acc_bl = np.array(acc_bl) * 100
    acc_kt = np.array(acc_kt) * 100

    plt.plot(x_points, acc_bl, color='pink', label='Baseline Condition')
    plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + all_modalities_type + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_fold_all_modalities(folds_proba_score_bl, folds_proba_score_bl2, folds_proba_score_kt, all_modalities_type,
                             title_name, xlabel, file_path, ylim=True, xticks=True):
    acc_bl = []
    acc_bl2 = []
    acc_kt = []
    x_points = []
    for num_obj in folds_proba_score_bl:
        x_points.append(num_obj)
        acc_bl.append(folds_proba_score_bl[num_obj][all_modalities_type]['test_acc'])
        acc_bl2.append(folds_proba_score_bl2[num_obj][all_modalities_type]['test_acc'])
        acc_kt.append(folds_proba_score_kt[num_obj][all_modalities_type]['test_acc'])
    acc_bl = np.array(acc_bl) * 100
    acc_bl2 = np.array(acc_bl2) * 100
    acc_kt = np.array(acc_kt) * 100

    plt.plot(x_points, acc_bl, color='pink', label='Baseline Condition')
    plt.plot(x_points, acc_bl2, color='red', label='Baseline 2 Condition')
    plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + all_modalities_type + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_each_modality(modalities_score, filename, title_name, xlabel, file_path, ylim=True, xticks=True):
    all_scores = {}
    x_points = []
    for num_obj in sorted(modalities_score):
        x_points.append(num_obj)
        for modality in modalities_score[num_obj]:
            all_scores.setdefault(modality, {'mean': [], 'std': []})
            all_scores[modality]['mean'].append(modalities_score[num_obj][modality]['mean'])
            all_scores[modality]['std'].append(modalities_score[num_obj][modality]['std'])
    # print('all_scores: ', all_scores)

    for modality in sorted(all_scores):
        all_scores[modality]['mean'] = np.array(all_scores[modality]['mean']) * 100
        all_scores[modality]['std'] = np.array(all_scores[modality]['std']) * 100
        plt.plot(x_points, all_scores[modality]['mean'], label=modality.capitalize())
        plt.fill_between(x_points, all_scores[modality]['mean'] - all_scores[modality]['std'],
                         all_scores[modality]['mean'] + all_scores[modality]['std'], alpha=0.3)
        # plt.errorbar(x=x_points, y=all_scores[modality]['mean'], yerr=all_scores[modality]['std'],
        #              fmt='-o', label=modality.capitalize())

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='upper left')
    plt.savefig(file_path + filename + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_all_modalities_v2(modalities_score_bl, modalities_score_kt, all_modalities_type,
                        title_name, xlabel, file_path, filename, ylim=True, xticks=True, errorbar=False,
                        plot_bl2=True):
    acc_bl = []
    std_bl = []
    acc_kt = []
    std_kt = []
    x_points = []
    for num_obj in sorted(modalities_score_bl):
        x_points.append(num_obj)
        acc_bl.append(modalities_score_bl[num_obj][all_modalities_type]['mean'])
        std_bl.append(modalities_score_bl[num_obj][all_modalities_type]['std'])
        acc_kt.append(modalities_score_kt[num_obj][all_modalities_type]['mean'])
        std_kt.append(modalities_score_kt[num_obj][all_modalities_type]['std'])
    acc_bl = np.array(acc_bl) * 100
    std_bl = np.array(std_bl) * 100
    acc_kt = np.array(acc_kt) * 100
    std_kt = np.array(std_kt) * 100
    # print('acc_bl, std_bl: ', acc_bl, std_bl)
    # print('acc_kt, std_kt: ', acc_kt, std_kt)

    if errorbar:
        plt.errorbar(x=x_points, y=acc_kt, yerr=std_kt, fmt='-x', color='#89bc73',
                     label='Transfer Condition (Trained on common latent features)')
        plt.errorbar(x=x_points, y=acc_bl, yerr=std_bl, fmt='-.o', color='#ea52bf',
                     label='Ground Truth Features (Trained on target context)')
    else:
        plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')
        plt.fill_between(x_points, acc_kt - std_kt, acc_kt + std_kt, color='blue', alpha=0.4)

        plt.plot(x_points, acc_bl, color='pink', label='Baseline 1 Condition')
        plt.fill_between(x_points, acc_bl - std_bl, acc_bl + std_bl, color='pink', alpha=0.4)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + filename + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_all_modalities(modalities_score_bl, modalities_score_bl2, modalities_score_kt, all_modalities_type,
                        title_name, xlabel, file_path, filename, ylim=True, xticks=True, errorbar=False,
                        plot_bl2=True):
    acc_bl = []
    std_bl = []
    acc_bl2 = []
    std_bl2 = []
    acc_kt = []
    std_kt = []
    x_points = []
    for num_obj in sorted(modalities_score_bl):
        x_points.append(num_obj)
        acc_bl.append(modalities_score_bl[num_obj][all_modalities_type]['mean'])
        std_bl.append(modalities_score_bl[num_obj][all_modalities_type]['std'])
        acc_bl2.append(modalities_score_bl2[num_obj][all_modalities_type]['mean'])
        std_bl2.append(modalities_score_bl2[num_obj][all_modalities_type]['std'])
        acc_kt.append(modalities_score_kt[num_obj][all_modalities_type]['mean'])
        std_kt.append(modalities_score_kt[num_obj][all_modalities_type]['std'])
    acc_bl = np.array(acc_bl) * 100
    std_bl = np.array(std_bl) * 100
    acc_bl2 = np.array(acc_bl2) * 100
    std_bl2 = np.array(std_bl2) * 100
    acc_kt = np.array(acc_kt) * 100
    std_kt = np.array(std_kt) * 100
    # print('acc_bl, std_bl: ', acc_bl, std_bl)
    # print('acc_kt, std_kt: ', acc_kt, std_kt)

    if errorbar:
        plt.errorbar(x=x_points, y=acc_kt, yerr=std_kt, fmt='-x', color='#89bc73',
                     label='Transfer Condition (Trained on common latent features)')
        plt.errorbar(x=x_points, y=acc_bl, yerr=std_bl, fmt='-.o', color='#ea52bf',
                     label='Ground Truth Features (Trained on target context)')
        if plot_bl2:
            plt.errorbar(x=x_points, y=acc_bl2, yerr=std_bl2, fmt='--D', color='#f18c5d',
                         label='Ground Truth Features (Trained on source context)')
    else:
        plt.plot(x_points, acc_kt, color='blue', label='Transfer Condition')
        plt.fill_between(x_points, acc_kt - std_kt, acc_kt + std_kt, color='blue', alpha=0.4)

        plt.plot(x_points, acc_bl, color='pink', label='Baseline 1 Condition')
        plt.fill_between(x_points, acc_bl - std_bl, acc_bl + std_bl, color='pink', alpha=0.4)

        if plot_bl2:
            plt.plot(x_points, acc_bl2, color='red', label='Baseline 2 Condition')
            plt.fill_between(x_points, acc_bl2 - std_bl2, acc_bl2 + std_bl2, color='red', alpha=0.4)

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('% Recognition Accuracy', fontsize=14)
    plt.title(title_name, fontsize=15)
    if ylim:
        plt.ylim(0, 100)
    if xticks:
        plt.xticks(x_points)
    plt.legend(loc='lower right')
    plt.savefig(file_path + filename + '.png', bbox_inches='tight', dpi=100)
    # plt.show()
    plt.close()


def plot_features_IE(features, features_y, dim_reduction, across_context, title, objects_labels, across, across_labels,
                     path, objects_to_skip=None):
    if objects_to_skip is None:
        objects_to_skip = []

    logging.info('features: {}'.format(features.shape))
    logging.info('features_y: {}'.format(features_y.shape, features_y.flatten()))

    dim_reduction_fn = get_dim_reduction_fn(dim_reduction)
    X_reduced = dim_reduction_fn.fit_transform(features)
    logging.info('X_reduced: {}'.format(X_reduced.shape))

    object_colors = ['orangered', 'blue', 'darkgreen', 'orange', 'maroon', 'lightblue', 'magenta', 'olive', 'brown',
                     'cyan', 'darkblue', 'beige', 'chartreuse', 'gold', 'green', 'grey', 'coral', 'black', 'khaki',
                     'orchid', 'steelblue', 'chocolate', 'indigo', 'crimson', 'fuchsia']
    markers = ['o', '^', 's', 'P', '*', 'X', 'D']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    fig.suptitle(title, fontsize='22')
    fig.subplots_adjust(top=0.92)  # title close to plot

    bbox_to_anchor_objects = (1.32, 1)

    legend_elements1 = []
    across_context_ = fix_names([across_context])[0]
    legend_elements1.append(Line2D([], [], marker=markers[across_labels[across_context]], color='w',
                                   label=across_context_, markerfacecolor='k', markersize=12))
    legend_elements2 = []
    for obj_name in objects_labels:
        if obj_name in objects_to_skip:
            continue
        obj_lab = objects_labels[obj_name]
        legend_elements2.append(mpatches.Patch(color=object_colors[obj_lab], label=obj_name))

    lab = across_labels[across_context]
    for obj_name in objects_labels:
        if obj_name in objects_to_skip:
            continue
        obj_lab = objects_labels[obj_name]
        indices = np.where((features_y == obj_lab))

        ax.scatter(X_reduced[indices, 0], X_reduced[indices, 1], c=object_colors[obj_lab], s=100,
                   edgecolor='black', marker=markers[lab])

    legend1 = ax.legend(handles=legend_elements1, title=across[:-1].capitalize() + ':', fontsize=12, title_fontsize=14,
                        loc='upper right')
    legend2 = ax.legend(handles=legend_elements2, title='Objects:', fontsize=12, title_fontsize=14,
                        bbox_to_anchor=bbox_to_anchor_objects, loc='upper right')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)

    file_name = '_'.join(title.split('-'))
    plt.savefig(path + os.sep + file_name, bbox_inches='tight', dpi=100)
    # plt.show(block=True)
    plt.close()


def plot_features_IE_v2(z1, y1, z2, y2, s_behavior, s_tool, t_behavior, t_tool, modality, objects_labels, across,
                        across_labels, path, objects_to_skip=None):
    if objects_to_skip is None:
        objects_to_skip = []

    logging.info('z1: {}'.format(z1.shape))
    logging.info('y1: {}'.format(y1.shape, y1.flatten()))
    logging.info('z2: {}'.format(z2.shape))
    logging.info('y2: {}'.format(y2.shape, y2.flatten()))

    object_colors = ['orangered', 'blue', 'darkgreen', 'orange', 'maroon', 'lightblue', 'magenta', 'olive', 'brown',
                     'cyan', 'darkblue', 'beige', 'chartreuse', 'gold', 'green', 'grey', 'coral', 'black', 'khaki',
                     'orchid', 'steelblue', 'chocolate', 'indigo', 'crimson', 'fuchsia']
    markers = ['o', '^', 's', 'P', '*', 'X', 'D']
    s_behavior_, s_tool_, t_behavior_, t_tool_, modality_ = fix_names(
        [s_behavior, s_tool, t_behavior, t_tool, modality])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    title = s_behavior_ + '-' + s_tool_ + '-' + modality_ + ' and ' + \
            t_behavior_ + '-' + t_tool_ + '-' + modality_ + '\n(Common Latent Features)'
    # For paper:
    # title = 'C) Shared Latent Feature Space'
    fig.suptitle(title, fontsize='22')
    fig.subplots_adjust(top=0.86)  # title close to plot
    # For paper:
    # fig.subplots_adjust(top=0.92)  # title close to plot

    bbox_to_anchor_objects = (1.32, 1)

    s_context, t_context = s_tool, t_tool
    if across == 'behaviors':
        s_context, t_context = s_behavior, t_behavior

    legend_elements1 = []
    for context in [s_context, t_context]:
        context_ = fix_names([context])[0]
        legend_elements1.append(Line2D([], [], marker=markers[across_labels[context]], color='w', label=context_,
                                       markerfacecolor='k', markersize=12))

    legend_elements2 = []
    for obj_name in objects_labels:
        if obj_name in objects_to_skip:
            continue
        obj_lab = objects_labels[obj_name]
        legend_elements2.append(mpatches.Patch(color=object_colors[obj_lab], label=obj_name))

    for obj_name in objects_labels:
        if obj_name in objects_to_skip:
            continue
        obj_lab = objects_labels[obj_name]
        indices = np.where((y1 == obj_lab))

        ax.scatter(z1[indices, 0], z1[indices, 1], c=object_colors[obj_lab], s=100, edgecolor='black',
                   marker=markers[across_labels[s_context]], alpha=0.8)
        ax.scatter(z2[indices, 0], z2[indices, 1], c=object_colors[obj_lab], s=100, edgecolor='black',
                   marker=markers[across_labels[t_context]], alpha=0.8)

    legend1 = ax.legend(handles=legend_elements1, title=across.capitalize() + ':', fontsize=12, title_fontsize=14,
                        loc='upper right')
    legend2 = ax.legend(handles=legend_elements2, title='Objects:', fontsize=12, title_fontsize=14,
                        bbox_to_anchor=bbox_to_anchor_objects, loc='upper right')
    # For paper:
    # legend2 = ax.legend(handles=legend_elements2, title='Objects:', fontsize=12, title_fontsize=14,
    #                     loc='upper left', framealpha=0.5)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)

    file_name = 'Latent_space_' + s_behavior_ + '_' + s_tool_ + '_' + modality_ + '_and_' + t_behavior_ + '_' + \
                t_tool_ + '_' + modality_
    plt.savefig(path + os.sep + file_name, bbox_inches='tight', dpi=100)
    # plt.show(block=True)
    plt.close()

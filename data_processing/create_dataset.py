# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import librosa
import numpy as np

from paper5.utils import get_config, read_images


def get_features(csv_file, usecols):
    data_ = np.genfromtxt(csv_file, delimiter=',', usecols=usecols)[1:]
    data_ = data_[~np.isnan(data_).any(axis=1)]  # Remove all rows with a nan

    return data_


def get_joint_state_data(modalities_idx_, joint_states_, num_joints_, data_info_, behavior_data_, dataset_metadata_):

    robot_ = data_info_['robot']
    behavior_ = data_info_['behavior']
    object_name_ = data_info_['object_name']
    tool_ = data_info_['tool']
    trial_ = data_info_['trial']

    for modality_ in modalities_idx_:
        joint_state_data = joint_states_[:, modalities_idx_[modality_]:modalities_idx_[modality_] + num_joints_]

        logging.info('modality: {} {} \n {}'.format(modality_, joint_state_data.shape, joint_state_data[:5]))
        # data_info_['modality'] = modality_

        behavior_data_[object_name_][tool_][trial_].setdefault(modality_, joint_state_data)

        metadata = {'frames': joint_state_data.shape[0], 'shape': joint_state_data.shape[1:],
                    'min': np.min(joint_state_data, axis=0), 'max': np.max(joint_state_data, axis=0)}
        logging.info('metadata: {}'.format(metadata))
        dataset_metadata_[behavior_][object_name_][tool_][trial_][modality_] = metadata

    return behavior_data_, dataset_metadata_


def dump_data(data_, dataset_path_, data_info_):
    robot_ = data_info_['robot']
    behavior_ = data_info_['behavior']
    object_name_ = data_info_['object_name']
    tool_ = data_info_['tool']
    trial_ = data_info_['trial']
    modality_ = data_info_['modality']

    dataset_path_ = os.sep.join([dataset_path_, robot_, behavior_, object_name_, tool_, trial_])
    os.makedirs(dataset_path_, exist_ok=True)
    db_file_name_ = dataset_path_ + os.sep + modality_ + '.bin'
    output_file_ = open(db_file_name_, 'wb')
    pickle.dump(data_, output_file_)
    output_file_.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a binary dataset from raw data.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset', 'Tool_Dataset_Prob'],
                        # required=True,
                        default='Tool_Dataset_Prob',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5', 'gen3-lite'],
                        default='ur5',
                        help='robot name')
    args = parser.parse_args()

    log_path = r'paper5/logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'paper5/logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    sensor_data_path = r'paper5/data' + os.sep + args.dataset
    dataset_path = r'paper5/data' + os.sep + args.dataset + '_Binary'

    config = get_config(r'paper5/configs' + os.sep + 'dataset_config.yaml')
    logging.info('config: {}'.format(config))

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities = config[args.dataset][args.robot]['modalities']
    logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    file_formats = ['.wav', '.jpg', '.csv']
    dump_image_data = False

    ds_metadata_full_filename = 'dataset_metadata_full.bin'
    ds_metadata_full_path = dataset_path + os.sep + args.robot + os.sep + ds_metadata_full_filename
    if os.path.exists(ds_metadata_full_path):
        logging.info('Loading dataset_metadata')
        bin_file = open(ds_metadata_full_path, 'rb')
        dataset_metadata = pickle.load(bin_file)
        bin_file.close()
    else:
        os.makedirs(dataset_path + os.sep + args.robot, exist_ok=True)
        dataset_metadata = {}

    for behavior in behaviors:
        behavior_data = {}
        for root, subdirs, files in os.walk(sensor_data_path, followlinks=True):
            for filename in files:
                filename, fileext = os.path.splitext(filename)

                if fileext in file_formats:
                    root_list = root.split(os.sep)
                    curr_behavior = root_list[-2]
                    if curr_behavior == behavior:
                        modality = root_list[-1]
                        trial = root_list[-3].split('_')[0]
                        object_name = root_list[-4]
                        tool = root_list[-5].split('_')[1]
                        robot = root_list[-5].split('_')[0]

                        data_info = {'robot': robot, 'behavior': behavior, 'modality': modality, 'trial': trial,
                                     'object_name': object_name, 'tool': tool}

                        logging.info('root: {}'.format(root))
                        logging.info('files: {}, {}'.format(len(files), files[:5]))
                        logging.info('filename: {}'.format(filename))
                        logging.info('curr_behavior: {}'.format(curr_behavior))
                        logging.info('modality: {}'.format(modality))
                        logging.info('trial: {}'.format(trial))
                        logging.info('object_name: {}'.format(object_name))
                        logging.info('tool: {}'.format(tool))
                        logging.info('robot: {}'.format(robot))

                        dataset_metadata.setdefault(behavior, {})

                        # Skipping processed data
                        dataset_temp_path = os.sep.join([dataset_path, robot, behavior, object_name, tool, trial])
                        modality_temp = modality
                        if fileext == '.csv':
                            if modality == 'joint_states':
                                modality_temp = 'position'
                            elif modality == 'wrench':
                                modality_temp = 'force'
                        db_temp_file_name = dataset_temp_path + os.sep + modality_temp + '.bin'
                        if os.path.exists(db_temp_file_name):
                            logging.info('{} already exists, so skipping ...'.format(db_temp_file_name))

                            # Restoring the metadata for vision data from old processing
                            if os.path.exists(ds_metadata_full_path):
                                bin_file = open(ds_metadata_full_path, 'rb')
                                dataset_metadata_temp = pickle.load(bin_file)
                                bin_file.close()

                                for mod in ['camera_depth_image', 'camera_rgb_image']:
                                    dataset_metadata[behavior].setdefault(object_name,
                                                                          {}).setdefault(tool, {}).setdefault(trial, {})
                                    dataset_metadata[behavior][object_name][tool][trial][mod] = \
                                        dataset_metadata_temp[behavior][object_name][tool][trial][mod]

                            break

                        # if trial not in ['trial-0'] or object_name not in ['chickpea'] or modality not in ['joint_states']:
                        #     continue

                        behavior_data.setdefault(object_name, {}).setdefault(tool, {}).setdefault(trial, {})
                        dataset_metadata[behavior].setdefault(object_name,
                                                              {}).setdefault(tool, {}).setdefault(trial, {})

                        if fileext == '.jpg':
                            data = read_images(root, files)
                            logging.info('data: {}'.format(data.shape))
                            behavior_data[object_name][tool][trial].setdefault(modality, data)
                            metadata = {'frames': data.shape[0], 'shape': data.shape[1:], 'min': np.min(data),
                                        'max': np.max(data)}
                            logging.info('metadata: {}'.format(metadata))
                            dataset_metadata[behavior][object_name][tool][trial][modality] = metadata
                        elif fileext == '.wav':
                            audio_time_series, sampling_rate = librosa.load(root + os.sep + files[0], sr=44100)
                            audio_length = len(audio_time_series) / sampling_rate

                            melspec = librosa.feature.melspectrogram(y=audio_time_series, sr=sampling_rate, n_fft=1024,
                                                                     hop_length=512, n_mels=60)
                            logspec = librosa.core.amplitude_to_db(melspec)
                            logspec = np.transpose(logspec)

                            logging.info('audio_length: {}'.format(audio_length))
                            logging.info('melspec: {}'.format(logspec.shape))

                            behavior_data[object_name][tool][trial].setdefault(modality, logspec)

                            metadata = {'frames': logspec.shape[0], 'shape': logspec.shape[1:], 'min': np.min(logspec),
                                        'max': np.max(logspec)}
                            logging.info('metadata: {}'.format(metadata))
                            dataset_metadata[behavior][object_name][tool][trial][modality] = metadata
                        elif fileext == '.csv':
                            if modality == 'joint_states':

                                num_joints = len(config[args.dataset][robot]['joints'])

                                if robot == 'ur5':
                                    joint_states = get_features(root + os.sep + files[0], usecols=range(9, 27))
                                    logging.info('joint_states: {} \n {}'.format(joint_states.shape, joint_states[:5]))

                                    modalities_idx = {'position': 0, 'velocity': 6, 'effort': 12}
                                    behavior_data, dataset_metadata = get_joint_state_data(modalities_idx, joint_states,
                                                                num_joints, data_info, behavior_data, dataset_metadata)
                                elif robot == 'gen3-lite':
                                    joint_states = get_features(root + os.sep + files[0], usecols=range(10, 29))
                                    logging.info('joint_states: {} \n {}'.format(joint_states.shape, joint_states[:5]))

                                    modalities_idx = {'position': 0, 'velocity': 7, 'effort': 14}
                                    behavior_data, dataset_metadata = get_joint_state_data(modalities_idx, joint_states,
                                                                num_joints, data_info, behavior_data, dataset_metadata)
                                else:
                                    assert False, 'Robot (' + robot + ') not supported!'
                            elif modality == 'gripper_joint_states':

                                if robot == 'ur5':
                                    # 4: position, 5: velocity
                                    gripper_state_data = get_features(root + os.sep + files[0], usecols=(4, 5))
                                elif robot == 'gen3-lite':
                                    continue
                                else:
                                    assert False, 'Robot (' + robot + ') not supported!'

                                logging.info('gripper_state_data: {} \n {}'.format(gripper_state_data.shape,
                                                                                   gripper_state_data[:5]))

                                behavior_data[object_name][tool][trial].setdefault(modality, gripper_state_data)
                                metadata = {'frames': gripper_state_data.shape[0], 'shape': gripper_state_data.shape[1:],
                                            'min': np.min(gripper_state_data, axis=0),
                                            'max': np.max(gripper_state_data, axis=0)}
                                logging.info('metadata: {}'.format(metadata))
                                dataset_metadata[behavior][object_name][tool][trial][modality] = metadata
                            elif modality == 'wrench':

                                if robot == 'ur5':
                                    wrench_states = get_features(root + os.sep + files[0], usecols=range(3, 9))

                                    logging.info('wrench_states: {} \n {}'.format(wrench_states.shape, wrench_states[:5]))

                                    modalities_idx = {'force': 0, 'torque': 3}
                                    for modality in modalities_idx:
                                        wrench_data = wrench_states[:, modalities_idx[modality]:
                                                                       modalities_idx[modality] + 3]
                                        logging.info('modality: {} {} \n {}'.format(modality, wrench_data.shape,
                                                                                    wrench_data[:5]))
                                        data_info['modality'] = modality

                                        behavior_data[object_name][tool][trial].setdefault(modality, wrench_data)
                                        metadata = {'frames': wrench_data.shape[0],
                                                    'shape': wrench_data.shape[1:],
                                                    'min': np.min(wrench_data, axis=0),
                                                    'max': np.max(wrench_data, axis=0)}
                                        logging.info('metadata: {}'.format(metadata))
                                        dataset_metadata[behavior][object_name][tool][trial][modality] = metadata
                                elif robot == 'gen3-lite':
                                    continue
                                else:
                                    assert False, 'Robot (' + robot + ') not supported!'
                            else:
                                assert False, 'Modality (' + modality + ') not supported'
                        else:
                            assert False, 'File extension (' + fileext + ') not supported'

                        logging.info('behavior_data[object_name][trial]: {}'.format(
                            set(behavior_data[object_name][tool][trial].keys())))

                        for modality in behavior_data[object_name][tool][trial].keys():
                            logging.info('Saving data: {}, {}, {}, {}, {}, {}'.format(robot, behavior, object_name,
                                                                                      tool, trial, modality))
                            data_info['modality'] = modality
                            if 'image' in modality and dump_image_data:
                                dump_data(behavior_data[object_name][tool][trial][modality], dataset_path, data_info)
                            elif 'image' not in modality:
                                dump_data(behavior_data[object_name][tool][trial][modality], dataset_path, data_info)
                        # clean after dump
                        modalities_dump = set(behavior_data[object_name][tool][trial].keys())
                        for modality in modalities_dump:
                            del behavior_data[object_name][tool][trial][modality]

                        output_file = open(ds_metadata_full_path, 'wb')
                        pickle.dump(dataset_metadata, output_file)
                        output_file.close()

                        logging.info('')
                        break
                    else:
                        break
                else:
                    break

    metadata_new = {}
    for behavior in dataset_metadata:
        logging.info('behavior: {}'.format(behavior))
        metadata_new.setdefault(behavior, {})
        metadata_new[behavior]['objects'] = set(dataset_metadata[behavior].keys())

        for object_name in dataset_metadata[behavior]:
            logging.info('object_name: {}'.format(object_name))
            metadata_new[behavior]['tools'] = set(dataset_metadata[behavior][object_name].keys())

            for tool in dataset_metadata[behavior][object_name]:
                logging.info('tool: {}'.format(tool))
                metadata_new[behavior]['trials'] = set(dataset_metadata[behavior][object_name][tool].keys())

                for trial in dataset_metadata[behavior][object_name][tool]:
                    logging.info('trial: {}'.format(trial))

                    for modality in dataset_metadata[behavior][object_name][tool][trial]:
                        logging.info('modality: {}, {}'.format(trial,
                                                               dataset_metadata[behavior][object_name][tool][trial][
                                                                   modality]))

                        metadata_new[behavior].setdefault('modalities', {})
                        metadata_new[behavior]['modalities'].setdefault(modality, {'avg_frames': [], 'shape': 0,
                                                                                   'min': [], 'max': []})
                        metadata_new[behavior]['modalities'][modality]['shape'] = \
                            dataset_metadata[behavior][object_name][tool][trial][modality]['shape']
                        metadata_new[behavior]['modalities'][modality]['avg_frames'].append(
                            dataset_metadata[behavior][object_name][tool][trial][modality]['frames'])
                        metadata_new[behavior]['modalities'][modality]['min'].append(
                            dataset_metadata[behavior][object_name][tool][trial][modality]['min'])
                        metadata_new[behavior]['modalities'][modality]['max'].append(
                            dataset_metadata[behavior][object_name][tool][trial][modality]['max'])

                        if metadata_new[behavior]['modalities'][modality]['shape']:
                            assert metadata_new[behavior]['modalities'][modality]['shape'] == \
                                   dataset_metadata[behavior][object_name][tool][trial][modality]['shape'], \
                                   'Size mismatch: {}, {}, {}, {}, {}: {} != {}'.format(behavior, object_name, tool,
                                    trial, modality, metadata_new[behavior]['modalities'][modality],
                                    dataset_metadata[behavior][object_name][tool][trial][modality])

        logging.info('metadata_new: {}'.format(metadata_new))

    # Computing average frames, min and max values over all examples
    for behavior in metadata_new:
        logging.info('behavior: {}'.format(behavior))
        for modality in metadata_new[behavior]['modalities']:
            logging.info('modality: {}'.format(modality))
            metadata_new[behavior]['modalities'][modality]['avg_frames'] = int(np.mean(metadata_new[behavior]['modalities'][modality]['avg_frames']))
            if isinstance(np.min(metadata_new[behavior]['modalities'][modality]['min']), (np.ndarray, np.generic)):
                metadata_new[behavior]['modalities'][modality]['min'] = np.min(metadata_new[behavior]['modalities'][modality]['min'], axis=0)
                metadata_new[behavior]['modalities'][modality]['max'] = np.max(metadata_new[behavior]['modalities'][modality]['max'], axis=0)
            else:
                metadata_new[behavior]['modalities'][modality]['min'] = np.min(metadata_new[behavior]['modalities'][modality]['min'])
                metadata_new[behavior]['modalities'][modality]['max'] = np.max(metadata_new[behavior]['modalities'][modality]['max'])

            logging.info('metadata_new: {}'.format(metadata_new[behavior]['modalities'][modality]))

    ds_metadata_filename = 'dataset_metadata.bin'
    ds_metadata_path = dataset_path + os.sep + args.robot + os.sep + ds_metadata_filename
    output_file = open(ds_metadata_path, 'wb')
    pickle.dump(metadata_new, output_file)
    output_file.close()

    logging.info('=============================:)=============================')

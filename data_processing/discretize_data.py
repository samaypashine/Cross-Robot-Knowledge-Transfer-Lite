# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import numpy as np
from skimage.transform import resize

from paper5.utils import get_config


def discretize_data(data_, discretize_temporal_bins_):

    frames = data_.shape[0]
    dimension = data_.shape[1]
    # Fix if number of frames is less than temporal_bins
    if frames < discretize_temporal_bins_:
        logging.info('{} is less than {} frames'.format(frames, discretize_temporal_bins_))
        data_ = resize(data_, (discretize_temporal_bins_, dimension))
        frames = data_.shape[0]
    size = frames // discretize_temporal_bins_

    discretized_data = []
    for a_bin in range(discretize_temporal_bins_):
        value = np.mean(data_[size * a_bin:size * (a_bin + 1)], axis=0)
        discretized_data.append(value)

    return np.array(discretized_data)


def discretize_data_v2(data_, discretize_temporal_bins_):

    frames = data_.shape[0]
    dimension = data_.shape[1]
    # Fix if number of frames is less than temporal_bins
    if frames < discretize_temporal_bins_:
        logging.info('{} is less than {} frames'.format(frames, discretize_temporal_bins_))
        data_ = resize(data_, (discretize_temporal_bins_, dimension))
        frames = data_.shape[0]
    frames_size = frames // discretize_temporal_bins_
    dimension_size = dimension // discretize_temporal_bins_

    discretized_data = []
    for a_bin in range(discretize_temporal_bins_):
        discretized_data2 = []
        for a_bin2 in range(discretize_temporal_bins_):
            value = np.mean(data_[frames_size * a_bin:frames_size * (a_bin + 1),
                            dimension_size * a_bin2:dimension_size * (a_bin2 + 1)])
            discretized_data2.append(value)
        discretized_data.append(discretized_data2)

    return np.array(discretized_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create a discretized binary dataset from binary data.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset', 'Tool_Dataset_Prob'],
                        # required=True,
                        default='Tool_Dataset_Prob',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5', 'gen3-lite'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-temporal-bins',
                        default=10,
                        type=int,
                        help='number of temporal bins')
    parser.add_argument('-normalization',
                        action='store_true',
                        help='normalization')
    args = parser.parse_args()

    log_path = r'paper5/logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'paper5/logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

    binary_dataset_path = r'paper5/data' + os.sep + args.dataset + '_Binary'
    discretize_dataset_path = r'paper5/data' + os.sep + args.dataset + '_Binary'  # + '_3mod'

    config = get_config(r'paper5/configs' + os.sep + 'dataset_config.yaml')
    logging.info('config: {}'.format(config))

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities = config[args.dataset][args.robot]['modalities']
    modalities = [modality for modality in modalities if 'image' not in modality]  # Skipping image modalities
    logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([binary_dataset_path, args.robot, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()
    logging.info('metadata: {}'.format(metadata))

    for root, subdirs, files in os.walk(binary_dataset_path):
        logging.info('root: {}'.format(root))
        for filename in files:
            filename, fileext = os.path.splitext(filename)

            if fileext != '.bin' or 'metadata' in filename:
                continue

            root_list = root.split(os.sep)
            behavior = root_list[-4]
            modality = filename
            trial = root_list[-1]
            object_name = root_list[-3]
            tool = root_list[-2]
            robot = root_list[-5]

            if modality not in modalities:
                continue

            logging.info('files: {}, {}'.format(len(files), files))
            logging.info('filename: {}'.format(filename))
            logging.info('behavior: {}'.format(behavior))
            logging.info('modality: {}'.format(modality))
            logging.info('trial: {}'.format(trial))
            logging.info('object_name: {}'.format(object_name))
            logging.info('tool: {}'.format(tool))
            logging.info('robot: {}'.format(robot))

            trial_data_path = root + os.sep + filename + '.bin'
            logging.info('trial_data_path: {}'.format(trial_data_path))

            data_file_path = trial_data_path
            bin_file = open(data_file_path, 'rb')
            data = pickle.load(bin_file)
            bin_file.close()

            logging.info('example: {}'.format(data.shape))

            if args.normalization:
                data_min = metadata[behavior]['modalities'][modality]['min']
                data_max = metadata[behavior]['modalities'][modality]['max']
                logging.info('data_min: {}'.format(data_min))
                logging.info('data_max: {}'.format(data_max))

                # Normalization
                norm_a = data - data_min
                norm_b = data_max - data_min
                delete_feature_idx = np.where(norm_b == 0)[0]
                # Deleting features where (data_max - data_min) = 0
                if len(delete_feature_idx) > 0:
                    if data.shape[-1] > len(delete_feature_idx):
                        logging.info('delete_feature_idx: {}'.format(delete_feature_idx))
                        data_min = np.delete(data_min, delete_feature_idx)
                        data_max = np.delete(data_max, delete_feature_idx)
                        data = np.delete(data, delete_feature_idx, axis=-1)
                        norm_a = data - data_min
                        norm_b = data_max - data_min
                data = np.divide(norm_a, norm_b)

                for i in delete_feature_idx:
                    data = np.insert(data, i, values=0, axis=1)  # Inserting 0 in column with same values

            if modality == 'audio':
                data = discretize_data_v2(data, args.temporal_bins)
            else:
                data = discretize_data(data, args.temporal_bins)

            logging.info('data: {}'.format(data.shape))
            # trial_dis_data_filepath = root + os.sep + modality + '-discretized-' + str(args.temporal_bins) + '-bins.bin'
            trial_dis_data_path = discretize_dataset_path + root.split(binary_dataset_path)[1] + os.sep
            trial_dis_data_filepath = trial_dis_data_path + modality + '-discretized-' + str(args.temporal_bins) + \
                                      ('-bins-norm.bin' if args.normalization else '-bins.bin')
            logging.info('trial_dis_data_filepath: {}'.format(trial_dis_data_filepath))

            os.makedirs(trial_dis_data_path, exist_ok=True)
            output_file = open(trial_dis_data_filepath, 'wb')
            pickle.dump(data, output_file)
            output_file.close()

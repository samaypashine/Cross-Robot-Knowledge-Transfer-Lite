# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import cv2
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from paper5.utils import get_config, fix_names


def plot_discretized_data(data_, data_info_, path_to_save=''):

    robot_ = data_info_['robot']
    behavior_ = fix_names([data_info_['behavior']])[0]
    object_name_ = data_info_['object_name']
    tool_ = fix_names([data_info_['tool']])[0]
    trial_ = data_info_['trial']
    modality_ = data_info_['modality']
    filename_ = data_info_['filename']

    x_values, y_values = data_.shape  # x: temporal bins
    logging.info('x_values, y_values: {}, {}'.format(x_values, y_values))

    title = robot_.capitalize() + '-' + behavior_.capitalize() + '-' + filename_.capitalize() + ' Features\n(' + \
            tool_.capitalize() + '-' + object_name_.capitalize() + ')'
    # For paper:
    # title = tool_ + '-' + behavior_ + '-' + fix_names([modality_])[0] + ' Features'
    plt.title(title, fontsize=16)
    plt.xlabel('Temporal Bins', fontsize=16)

    if modality_ in ['effort', 'position', 'velocity', 'gripper_joint_states']:
        im = plt.imshow(data_.T, cmap='GnBu')
        y_label = 'Joints'
    elif modality_ in ['force', 'torque']:
        im = plt.imshow(data_.T, cmap='GnBu')
        y_label = 'Axis'
    elif modality_ in ['audio']:
        im = plt.imshow(np.flipud(data_.T), cmap='GnBu')
        y_label = 'Frequency Bins'
    else:
        y_label = ''
    plt.ylabel(y_label, fontsize=16)

    ax = plt.gca()
    ax.set_xticks(np.arange(0, x_values, 1))
    ax.set_xticklabels(np.arange(1, x_values + 1, 1))
    if modality_ in ['force', 'torque']:
        axis = ['x', 'y', 'z']
        ax.set_yticks(np.arange(0, len(axis), 1))
        ax.set_yticklabels(axis)
    else:
        ax.set_yticks(np.arange(0, y_values, 1))
        ax.set_yticklabels(np.arange(1, y_values + 1, 1))

    # plt.colorbar(im, fraction=0.02, pad=0.04)
    # Colorbar with same height as the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.1)
    plt.colorbar(im, cax=cax)

    if path_to_save:
        file_path = path_to_save + os.sep + filename_ + '.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=100)

    # plt.show()
    plt.close()


def plot_audio(data_, data_info_, path_to_save=''):

    robot_ = data_info_['robot']
    behavior_ = data_info_['behavior']
    object_name_ = data_info_['object_name']
    tool_ = data_info_['tool']
    trial_ = data_info_['trial']
    modality_ = data_info_['modality']
    filename_ =data_info_['filename']

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(data_.T, hop_length=512, x_axis='s', y_axis='linear', ax=ax, sr=44100, cmap='magma')
    title = robot_.capitalize() + '-' + behavior_.capitalize() + '-' + filename_.capitalize() + ' Raw Signal\n(' + \
            tool_.capitalize() + '-' + object_name_.capitalize() + ')'
    fig.suptitle(title, fontsize=16)
    fig.colorbar(img, ax=ax, format='%+2.f dB')
    ax.xaxis.set_label_text('Time (Seconds)', fontsize=16)
    ax.yaxis.set_label_text('Hz', fontsize=16)

    if path_to_save:
        file_path = path_to_save + os.sep + filename_ + '.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=100)

    # plt.show()
    plt.close()


def plot_joint_data(data_, data_info_, path_to_save=''):

    robot_ = data_info_['robot']
    behavior_ = data_info_['behavior']
    object_name_ = data_info_['object_name']
    tool_ = data_info_['tool']
    trial_ = data_info_['trial']
    modality_ = data_info_['modality']
    filename_ = data_info_['filename']

    title = robot_.capitalize() + '-' + behavior_.capitalize() + '-' + filename_.capitalize() + ' Raw Signal\n(' + \
            tool_.capitalize() + '-' + object_name_.capitalize() + ')'

    if modality_ in ['force', 'torque']:
        axis = ['x', 'y', 'z']
        for i_ in range(len(axis)):
            plt.plot(data_[:, i_], label=axis[i_])
        plt.legend(title='Axis', loc='upper right')
    else:
        joints = data_.shape[1]
        for joint_idx in range(joints):
            plt.plot(data_[:, joint_idx], label=str(joint_idx + 1))
        plt.legend(title='Joints', loc='upper right')

    plt.title(title, fontsize=16)
    plt.xlabel('Samples', fontsize=16)
    plt.ylabel(modality_.capitalize(), fontsize=16)

    if path_to_save:
        file_path = path_to_save + os.sep + filename_ + '.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=100)

    # plt.show()
    plt.close()


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
    args = parser.parse_args()

    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    plot_dataset_path = r'data' + os.sep + args.dataset + '_Plot'

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

    for root, subdirs, files in os.walk(binary_dataset_path):
        logging.info('root: {}'.format(root))
        for filename in files:
            filename, fileext = os.path.splitext(filename)

            if fileext != '.bin' or 'metadata' in filename:
                continue

            root_list = root.split(os.sep)
            behavior = root_list[-4]
            modality = '-'.join(filename.split('-')[:1])
            trial = root_list[-1]
            object_name = root_list[-3]
            tool = root_list[-2]
            robot = root_list[-5]

            data_info = {'robot': robot, 'behavior': behavior, 'modality': modality, 'trial': trial,
                         'object_name': object_name, 'tool': tool, 'filename': filename}

            logging.info('files: {}, {}'.format(len(files), files))
            logging.info('filename: {}'.format(filename))
            logging.info('behavior: {}'.format(behavior))
            logging.info('modality: {}'.format(modality))
            logging.info('trial: {}'.format(trial))
            logging.info('object_name: {}'.format(object_name))
            logging.info('tool: {}'.format(tool))
            logging.info('robot: {}'.format(robot))

            # if filename not in ['audio', 'audio-discretized-10-bins', 'audio-autoencoder-linear-tl',
            #                     'effort', 'effort-discretized-10-bins', 'effort-autoencoder-linear-tl',
            #                     'force', 'force-discretized-10-bins', 'force-autoencoder-linear-tl'] \
            #         or behavior not in ['3-stirring-fast', '6-poke'] \
            #         or tool not in ['metal-scissor', 'plastic-spoon']:
            #     continue

            trial_data_path = root + os.sep + filename + '.bin'
            logging.info('trial_data_path: {}'.format(trial_data_path))

            bin_file = open(trial_data_path, 'rb')
            data = pickle.load(bin_file)
            bin_file.close()

            if 'autoencoder' in filename:
                logging.info('data_gen: {}'.format(data['data_gen'].shape))
                logging.info('code: {}'.format(data['code'].shape))
            else:
                logging.info('data: {}'.format(data.shape))

            plot_path = os.sep.join([plot_dataset_path, robot, behavior, object_name, tool, trial])
            os.makedirs(plot_path, exist_ok=True)
            logging.info('plot_path: {}'.format(plot_path))

            if 'discretized' in filename:
                plot_discretized_data(data, data_info, path_to_save=plot_path)
            elif 'autoencoder' in filename:
                data = data['data_gen']
                data_min = metadata[behavior]['modalities'][modality]['min']
                data_max = metadata[behavior]['modalities'][modality]['max']
                logging.info('data_min: {}'.format(data_min))
                logging.info('data_max: {}'.format(data_max))

                norm_b = data_max - data_min
                delete_feature_idx = np.where(norm_b == 0)[0]
                # Deleting features where (data_max - data_min) = 0
                if len(delete_feature_idx) > 0:
                    if data.shape[-1] > len(delete_feature_idx):
                        logging.info('delete_feature_idx: {}'.format(delete_feature_idx))
                        data_min = np.delete(data_min, delete_feature_idx)
                        data_max = np.delete(data_max, delete_feature_idx)
                        norm_b = data_max - data_min
                data = (data * norm_b) + data_min

                if 'image' in modality:
                    img = cv2.convertScaleAbs(data)
                    # Quality for JPEG encoding in range 1-100
                    cv2.imwrite(plot_path + os.sep + f'{filename}' + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 100])
                elif 'audio' in modality:
                    plot_audio(data, data_info, path_to_save=plot_path)
                else:
                    plot_joint_data(data, data_info, path_to_save=plot_path)
            else:
                if modality == 'audio':
                    plot_audio(data, data_info, path_to_save=plot_path)
                elif 'image' in modality:
                    if 'last-image' in filename:
                        cv2.imwrite(plot_path + os.sep + f'{filename}' + '.jpg', data, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    else:
                        folder_path = plot_path + os.sep + modality
                        os.makedirs(folder_path, exist_ok=True)
                        for i in range(len(data)):
                            # Quality for JPEG encoding in range 1-100
                            cv2.imwrite(folder_path + os.sep + f'{i:05}' + '.jpg', data[i], [cv2.IMWRITE_JPEG_QUALITY, 80])
                else:
                    plot_joint_data(data, data_info, path_to_save=plot_path)

# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import numpy as np
import scipy.io
from scipy.io import loadmat

from paper5.utils import get_config, get_split_data_objects, get_classes_labels, get_dim_reduction_fn, plot_features_IE, \
    plot_features_IE_v2, fix_names
import matlab.engine


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot the dataset from binary data.')
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
    parser.add_argument('-dim-reduction',
                        choices=['PCA', 'ISOMAP', 'TSNE'],
                        default='PCA',
                        help='dimensionality reduction function')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_path = r'logs' + os.sep + time_stamp + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    logging.info('config: {}'.format(config))

    robots = list(config[args.dataset].keys())
    behaviors = config[args.dataset][args.robot]['behaviors']
    modalities = config[args.dataset][args.robot]['modalities']
    logging.info('robots: {}'.format(robots))
    logging.info('behaviors: {}'.format(behaviors))
    logging.info('modalities: {}'.format(modalities))
    logging.info('modalities: {}'.format(modalities))

    data_file_path = os.sep.join([binary_dataset_path, args.robot, 'dataset_metadata.bin'])
    bin_file = open(data_file_path, 'rb')
    metadata = pickle.load(bin_file)
    bin_file.close()
    # logging.info('metadata: {}'.format(metadata))

    objects = metadata[behaviors[0]]['objects']
    tools = metadata[behaviors[0]]['tools']
    trials = metadata[behaviors[0]]['trials']
    logging.info('objects: {}'.format(objects))
    logging.info('tools: {}'.format(tools))
    logging.info('trials: {}'.format(trials))

    across = 'robots'

    projections = []
    for source_robot in sorted(robots):
        for target_robot in sorted(robots):
            for source_behavior in sorted(behaviors):
                for source_tool in sorted(tools):
                    for target_behavior in sorted(behaviors):
                        for target_tool in sorted(tools):
                            # Both behaviors and tools cannot be the same
                            if (source_robot != target_robot) and (source_behavior == target_behavior) and (source_tool == target_tool):

                                # exists = False
                                # for projection in projections:
                                #     source_robot_ = projection['source_robot']
                                #     source_behavior_ = projection['source_behavior']
                                #     source_tool_ = projection['source_tool']
                                #     target_robot_ = projection['target_robot']
                                #     target_behavior_ = projection['target_behavior']
                                #     target_tool_ = projection['target_tool']
                                #     if (source_robot_ != target_robot_) and (source_behavior_ == target_behavior) and (source_tool_ == target_tool_):
                                #         exists = True
                                #         break

                                # if not exists:
                                    # Either behavior or tool needs to be the same
                                # if source_robot != target_robot and source_behavior == target_behavior and source_tool == target_tool:
                                projections.append({'source_robot': source_robot, 'source_behavior': source_behavior, 'source_tool': source_tool,
                                                    'target_robot': target_robot, 'target_behavior': target_behavior, 'target_tool': target_tool})
                                    # elif source_behavior != target_behavior and source_tool == target_tool:
                                    #     projections.append({'source_robot': source_robot, 'source_behavior': source_behavior, 'source_tool': source_tool,
                                    #                         'target_robot': target_robot, 'target_behavior': target_behavior, 'target_tool': target_tool})
    logging.info('projections: {}'.format(len(projections)))

    dim_reduction_fn = get_dim_reduction_fn(args.dim_reduction)

    MATLAB_eng = matlab.engine.start_matlab()
    MATLAB_eng.cd(r'paper5' + os.sep + 'kema', nargout=0)

    plot_dataset_path = r'data' + os.sep + args.dataset + '_Plot_Features' + os.sep + 'IE' + os.sep
    data_path_KEMA = os.getcwd() + os.sep + plot_dataset_path + os.sep + 'KEMA_data' + os.sep + time_stamp
    input_filename_KEMA = 'data.mat'
    output_filename_KEMA = 'projections.mat'
    os.makedirs(data_path_KEMA, exist_ok=True)

    objects_labels = get_classes_labels(metadata[behaviors[0]]['objects'])
    logging.info('objects_labels: {}'.format(objects_labels))

    for p_i, projection in enumerate(projections):

        source_robot = projection['source_robot']
        source_behavior = projection['source_behavior']
        source_tool = projection['source_tool']
        target_robot = projection['target_robot']
        target_behavior = projection['target_behavior']
        target_tool = projection['target_tool']

        # 1-look, 2-stirring-slow, 3-stirring-fast, 4-stirring-twist, 5-whisk, 6-poke
        # wooden-fork, plastic-knife, plastic-spoon, metal-whisk, metal-scissor, wooden-chopstick
        # source_behavior = '2-stirring-slow'
        # source_tool = 'plastic-spoon'
        # target_behavior = '3-stirring-fast'
        # target_tool = 'plastic-spoon'

        # if '1-look' in [source_behavior, target_behavior]:
        #     continue

        # across = 'tools'
        # if source_behavior != target_behavior:
        #     across = 'behaviors'
        logging.info('across: {}'.format(across))

        # across_labels = get_classes_labels(metadata[behaviors[0]]['robots'])
        across_labels = get_classes_labels(list(config[args.dataset].keys()))
        # across_labels = get_classes_labels(metadata[behaviors[0]]['tools'])
        # if across == 'behaviors':
        #     across_labels = get_classes_labels(behaviors)
        logging.info('across_labels: {}'.format(across_labels))

        source_robot_, source_behavior_, source_tool_, target_robot_, target_behavior_, target_tool_ = fix_names([source_robot, source_behavior, source_tool,
                                                                                                                  target_robot, target_behavior, target_tool])
        
        projection_dir = '_'.join([source_robot_, source_behavior_, source_tool_, 'and', target_robot_, target_behavior_, target_tool_])
        projection_path = plot_dataset_path + projection_dir
        os.makedirs(projection_path, exist_ok=True)

        for modality in modalities:
            logging.info('source_robot: {}'.format(source_robot))
            logging.info('source_behavior: {}'.format(source_behavior))
            logging.info('source_tool: {}'.format(source_tool))
            logging.info('target_robot: {}'.format(target_robot))
            logging.info('target_behavior: {}'.format(target_behavior))
            logging.info('target_tool: {}'.format(target_tool))
            logging.info('modality: {}'.format(modality))

            # Get all objects to train the projection function
            objects_list = list(objects_labels.keys())

            # Get source train objects data
            s_data, s_y = get_split_data_objects(binary_dataset_path, trials, objects_labels, source_robot,
                                                 source_behavior, modality, source_tool, objects_list, args.feature, 'resnet18')
            logging.info('s_data: {}'.format(s_data.shape))
            logging.info('s_y: {}, {}'.format(s_y.shape, s_y.flatten()[0:15]))

            # Get target train objects data
            t_data, t_y = get_split_data_objects(binary_dataset_path, trials, objects_labels, target_robot,
                                                 target_behavior, modality, target_tool, objects_list, args.feature, 'resnet18')
            logging.info('t_data: {}'.format(t_data.shape))
            logging.info('t_y: {}, {}'.format(t_y.shape, t_y.flatten()[0:15]))

            # Transfer Robot Knowledge
            # KEMA
            KEMA_data = {'X1': s_data, 'Y1': s_y + 1, 'X2': t_data, 'Y2': t_y + 1,
                         'X2_Test': t_data}  # add 1 as in KEMA (MATLAB) labels starts from 1

            scipy.io.savemat(os.path.join(data_path_KEMA, input_filename_KEMA), mdict=KEMA_data)
            MATLAB_eng.project2Domains_v2(data_path_KEMA, input_filename_KEMA, output_filename_KEMA, 1)

            # In case Matlab messes up, we'll load and check these immediately, then delete them so we never read in an old file
            projections = None
            if os.path.isfile(os.path.join(data_path_KEMA, output_filename_KEMA)):
                try:
                    projections = loadmat(os.path.join(data_path_KEMA, output_filename_KEMA))
                    Z1, Z2, Z2_Test = projections['Z1'], projections['Z2'], projections['Z2_Test']
                    os.remove(os.path.join(data_path_KEMA, output_filename_KEMA))
                    os.remove(os.path.join(data_path_KEMA, input_filename_KEMA))
                except TypeError as e:
                    logging.info('loadmat failed: {}'.format(e))

            logging.info('Z1: {}'.format(Z1.shape))
            logging.info('Z2: {}'.format(Z2.shape))
            logging.info('Z2_Test: {}'.format(Z2_Test.shape))

            if Z1.shape[1] > 1:
                # wheat, chia-seed, glass-bead, styrofoam-bead, metal-nut-bolt, split-green-pea, detergent, salt,
                # kidney-bean, empty, plastic-bead, wooden-button, water, cane-sugar, chickpea
                objects_to_skip = []
                # For paper:
                # objects_to_skip = ['metal-nut-bolt', 'cane-sugar', 'wooden-button', 'wheat', 'water', 'empty',
                #                    'styrofoam-bead', 'split-green-pea', 'glass-bead']

                modality_ = fix_names([modality])[0]
                across_context = source_tool
                if across == 'behaviors':
                    across_context = source_behavior
                elif across == 'robots':
                    across_context = source_robot
                title1 = source_robot_ + '-' + source_behavior_ + '-' + source_tool_ + '-' + modality_
                # For paper:
                # title1 = 'A) ' + source_tool_ + '-' + source_behavior_ + '-' + modality_ + ' Feature Space'
                plot_features_IE(s_data, s_y, args.dim_reduction, across_context, title1, objects_labels, across,
                                 across_labels, projection_path, objects_to_skip)

                across_context = target_tool
                if across == 'behaviors':
                    across_context = target_behavior
                elif across == 'robots':
                    across_context = target_robot
                title2 = target_behavior_ + '-' + target_tool_ + '-' + modality_
                # For paper:
                # title2 = 'B) ' + target_tool_ + '-' + target_behavior_ + '-' + modality_ + ' Feature Space'
                plot_features_IE(t_data, t_y, args.dim_reduction, across_context, title2, objects_labels, across,
                                 across_labels, projection_path, objects_to_skip)

                plot_features_IE_v2(Z1, s_y, Z2, t_y, source_robot, source_behavior, source_tool, target_robot, target_behavior, target_tool,
                                    modality, objects_labels, across, across_labels, projection_path, objects_to_skip)

        # exit()

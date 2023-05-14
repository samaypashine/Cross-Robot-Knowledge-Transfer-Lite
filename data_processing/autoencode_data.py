# Author: Gyan Tatiya

import argparse
import os
import logging
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

# from sequitur.models import LINEAR_AE, LSTM_AE, CONV_LSTM_AE
# from sequitur import quick_train

from paper5.utils import get_config
from paper5.pytorch import models
from paper5.pytorch.utils import EDNTrainer, EDNDataset


if __name__ == '__main__':
    """
    Assumptions:
    Only using Linear autoencoder
    For image modalities:
        Use only the last image in the video
        Reduce height and width of the image
    For audio, reduce time frames
    For all modalities, normalize the data and use ReLU for hidden layers and Sigmoid for last layer
    """

    parser = argparse.ArgumentParser(description='Create a autoencoder binary dataset from binary data.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset_2Tools_2Contents', 'Tool_Dataset'],
                        # required=True,
                        default='Tool_Dataset',
                        help='dataset name')
    parser.add_argument('-robot',
                        choices=['ur5'],
                        default='ur5',
                        help='robot name')
    parser.add_argument('-autoencoder',
                        choices=['Linear'],
                        default='Linear',
                        help='autoencoder type')
    args = parser.parse_args()

    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    logging.info('args: {}'.format(args))

    binary_dataset_path = r'data' + os.sep + args.dataset + '_Binary'
    output_binary_dataset_path = binary_dataset_path  # + '_AE_Z'

    config = get_config(r'configs' + os.sep + 'dataset_config.yaml')
    model_config = get_config(r'configs' + os.sep + 'autoencoder_config.yaml')

    logging.info('config: {}'.format(config))
    logging.info('model_config: {}'.format(model_config))

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

    # Hyper-Parameters
    hidden_layer_units = model_config[args.autoencoder]['hyper_parameters']['hidden_layer_units']
    code_vector = model_config[args.autoencoder]['hyper_parameters']['code_vector']

    training_epochs = model_config[args.autoencoder]['trainer']['training_epochs']
    learning_rate = model_config[args.autoencoder]['trainer']['learning_rate']
    use_triplet_loss = model_config[args.autoencoder]['trainer']['use_triplet_loss']

    repr_code = 'autoencoder-linear-tl' if use_triplet_loss else 'autoencoder-linear'

    for behavior in behaviors:
        logging.info('behavior: {}'.format(behavior))
        for modality in modalities:
            logging.info('modality: {}'.format(modality))

            dataset = EDNDataset(binary_dataset_path, args.robot, metadata, behavior, modality, use_triplet_loss)
            logging.info('dataset: {}, {}'.format(len(dataset), dataset.data_filepaths[:2]))
            logging.info('delete_feature_idx: {}'.format(dataset.delete_feature_idx))
            logging.info('dataset.shape: {}'.format(dataset.shape))
            logging.info('dataset.num_features: {}'.format(dataset.num_features))
            logging.info('dataset.dataset_shape: {}'.format(dataset.dataset_shape))

            if not dataset.bad_data:
                data_temp = dataset[0][0].cpu().numpy()
                logging.info('data_temp: {}, {}, {}'.format(data_temp.shape, type(data_temp), data_temp.dtype))
                logging.info('data_temp: \n{}'.format(np.round(data_temp[:30], 2)))

            model_class = getattr(models, model_config[args.autoencoder]['model'])

            logging.info('model_class: {}'.format(model_class))

            model = model_class(dataset.num_features, dataset.num_features, hidden_layer_sizes=hidden_layer_units,
                                n_dims_code=code_vector, h_activation_fn=nn.ReLU(), out_activation_fn=nn.Sigmoid())
            logging.info('model: \n{}'.format(model))

            batch_size = len(dataset)
            if 'image' in modality or 'audio' in modality:
                batch_size = 1

            trainer = EDNTrainer(dataset, model=model, lr=learning_rate, batch_size=batch_size)
            cost_log = trainer.train_model(training_epochs)

            encodings = np.array([trainer.generate_code(s[0]).cpu().numpy() if dataset.use_triplet_loss else
                                  trainer.generate_code(s).cpu().numpy() for s, t in dataset])
            logging.info('encodings: {}\n{}'.format(encodings.shape, encodings[0][:30]))

            data_gen = np.array([trainer.generate_output(s[0]).cpu().numpy() if dataset.use_triplet_loss else
                                 trainer.generate_output(s).cpu().numpy() for s, t in dataset])
            logging.info('data_gen: {}, {}, {}'.format(data_gen.shape, type(data_gen), data_gen.dtype))
            logging.info('data_gen: \n{}'.format(data_gen[0][:30]))

            data_gen = data_gen.reshape(dataset.dataset_shape)
            logging.info('data_gen: {}, {}, {}'.format(data_gen.shape, type(data_gen), data_gen.dtype))

            # plt.plot(range(1, len(cost_log) + 1), cost_log)
            # plt.show()
            # plt.close()

            # im = plt.imshow(data_gen[0])
            # plt.show()
            # plt.close()

            for tool in dataset.tools_objects_trials_idx:
                for object_name in dataset.tools_objects_trials_idx[tool]:
                    for trial_num in dataset.tools_objects_trials_idx[tool][object_name]:
                        data = {'code': encodings[dataset.tools_objects_trials_idx[tool][object_name][trial_num]],
                                'data_gen': data_gen[dataset.tools_objects_trials_idx[tool][object_name][trial_num]]}
                        trial_data_path = os.sep.join([output_binary_dataset_path, args.robot, behavior, object_name,
                                                       tool, trial_num])
                        os.makedirs(trial_data_path, exist_ok=True)
                        output_file = open(trial_data_path + os.sep + modality + '-' + repr_code + '.bin', 'wb')
                        pickle.dump(data, output_file)
                        output_file.close()

            '''
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_list = to_tensor(data_list).to(device, dtype=torch.float)

            # Hyper-Parameters
            TRAINING_EPOCHS = 1000
            LEARNING_RATE = 0.0001
            CODE_VECTOR = 125
            HIDDEN_LAYER_UNITS = [1000, 500, 250]
            ACTIVATION_FUNCTION = nn.Sigmoid()  # nn.ELU()

            encoder, decoder, encodings, losses = quick_train(LINEAR_AE, data_list, encoding_dim=CODE_VECTOR,
                                                              h_dims=HIDDEN_LAYER_UNITS, h_activ=ACTIVATION_FUNCTION,
                                                              out_activ=ACTIVATION_FUNCTION, denoise=True)

            # CONV_LSTM_AE - works for images, not videos
            # encoder, decoder, encodings, losses = quick_train(CONV_LSTM_AE, data_list, encoding_dim=2, denoise=True,
            #                                                   kernel=(3, 3), stride=(3, 3))

            print('encoder: ', encoder)
            print('decoder: ', decoder)
            print('encodings: ', len(encodings), encodings)
            print('encodings: ', len(encodings), encodings[0].shape)
            print('losses: ', len(losses), losses)
            exit()
            '''

            '''
            TODO:
            X Add GPU usage
            Print less loss for epochs
            
            X Load data in Dataset Class not before
            '''

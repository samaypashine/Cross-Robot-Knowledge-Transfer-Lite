# Author: Gyan Tatiya

import logging
import os
import pickle
import subprocess

import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from paper5.utils import read_images, add_nose_time_shift, add_noise_salt_pepper


# Utils #

def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v).float()
    else:
        return torch.tensor(v, dtype=torch.float)


def get_gpu_memory_map(max_memory_used_percent=0.8):
    """Get the current gpu usage if used memory is less than max_memory_used_percent
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    memory_used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                          encoding='utf-8')
    memory_total = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
                                           encoding='utf-8')
    memory_used = np.array([int(x) for x in memory_used.strip().split('\n')])
    memory_total = np.array([int(x) for x in memory_total.strip().split('\n')])

    memory_used_percent = memory_used / memory_total
    logging.info('{} GPU memory usage: {}'.format(len(memory_used_percent), [round(m, 2) for m in memory_used_percent]))

    gpu_memory_map = {}
    for gpu_id in range(len(memory_used)):
        if memory_used_percent[gpu_id] < max_memory_used_percent:
            gpu_memory_map[gpu_id] = memory_used[gpu_id]

    return gpu_memory_map


# Losses #

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


# Datasets #

class EDNDataset(Dataset):
    def __init__(self, dataset_path, robot, metadata, behavior, modality, use_triplet_loss=False):
        self.metadata = metadata
        self.behavior = behavior
        self.modality = modality
        self.use_triplet_loss = use_triplet_loss

        objects = self.metadata[behavior]['objects']
        tools = self.metadata[behavior]['tools']
        trials = self.metadata[behavior]['trials']

        self.data_filepaths = []
        self.tools_objects_trials_idx = {}
        i = 0
        for tool in tools:
            self.tools_objects_trials_idx.setdefault(tool, {})
            for object_name in sorted(objects):
                self.tools_objects_trials_idx[tool].setdefault(object_name, {})
                for trial_num in sorted(trials):
                    if 'image' in modality:
                        trial_data_filepath = os.sep.join([dataset_path, robot, behavior, object_name, tool, trial_num,
                                                           modality + '-last-image.bin'])
                        if not os.path.exists(trial_data_filepath):
                            trial_data_filepath = os.sep.join([dataset_path, robot, behavior, object_name, tool,
                                                               trial_num, modality + '.bin'])
                            if not os.path.exists(trial_data_filepath):
                                raw_dataset_path = '_'.join(dataset_path.split('_')[:-1])
                                trials_timestamp = os.listdir(os.sep.join([raw_dataset_path, robot + '_' + tool,
                                                                           object_name]))
                                for trial_timestamp in trials_timestamp:
                                    trial = trial_timestamp.split('_')[0]
                                    if trial == trial_num:
                                        trial_num_timestamp = trial_timestamp
                                        break

                                trial_data_path = os.sep.join([raw_dataset_path, robot + '_' + tool, object_name,
                                                               trial_num_timestamp, behavior, modality])
                                files = sorted(os.listdir(trial_data_path))
                                data = read_images(trial_data_path, [files[-1]])[0]  # Use last image
                            else:
                                bin_file = open(trial_data_filepath, 'rb')
                                data = pickle.load(bin_file)[-1]  # Use last image
                                bin_file.close()

                            trial_data_filepath = os.sep.join([dataset_path, robot, behavior, object_name, tool,
                                                               trial_num, modality + '-last-image.bin'])
                            output_file = open(trial_data_filepath, 'wb')
                            pickle.dump(data, output_file)
                            output_file.close()
                    else:
                        trial_data_filepath = os.sep.join([dataset_path, robot, behavior, object_name, tool, trial_num,
                                                           modality + '.bin'])

                    self.data_filepaths.append(trial_data_filepath)

                    self.tools_objects_trials_idx[tool][object_name][trial_num] = i
                    i += 1

        self.shape = list(self.metadata[self.behavior]['modalities'][self.modality]['shape'])
        if 'image' in self.modality:
            self.shape[0] = self.shape[0] // 4  # reducing image height
            self.shape[1] = self.shape[1] // 4  # reducing image width
        elif 'audio' in self.modality:
            self.shape[0] = self.shape[0] // 4  # reducing time frames
            self.shape.insert(0, self.metadata[self.behavior]['modalities'][self.modality]['avg_frames'])
        else:
            self.shape.insert(0, self.metadata[self.behavior]['modalities'][self.modality]['avg_frames'])

        self.num_features = np.prod(self.shape)
        self.dataset_shape = [len(self.data_filepaths)] + self.shape

        self.data_min = self.metadata[self.behavior]['modalities'][self.modality]['min']
        self.data_max = self.metadata[self.behavior]['modalities'][self.modality]['max']
        logging.info('data_min: {}'.format(self.data_min))
        logging.info('data_max: {}'.format(self.data_max))

        self.norm_b = self.data_max - self.data_min

        idx = np.where(self.norm_b == 0)[0]
        # Deleting features where (data_max - data_min) = 0
        self.delete_feature_idx = []
        self.bad_data = False
        if len(idx) > 0:
            logging.info('delete_feature_idx: {}'.format(idx))
            if self.shape[-1] > len(idx):
                self.delete_feature_idx = idx
                self.data_min = np.delete(self.data_min, self.delete_feature_idx)
                self.data_max = np.delete(self.data_max, self.delete_feature_idx)
                self.norm_b = self.data_max - self.data_min
                logging.info('data_min: {}'.format(self.data_min))
                logging.info('data_max: {}'.format(self.data_max))

                self.shape[-1] = self.shape[-1] - len(self.delete_feature_idx)
                self.num_features = np.prod(self.shape)
                self.dataset_shape = [len(self.data_filepaths)] + self.shape
            else:
                self.bad_data = True
                logging.info('bad_data (All the features are constant!): {}'.format(self.bad_data))

    def __len__(self):
        return len(self.data_filepaths)

    def get_data(self, item):

        bin_file = open(self.data_filepaths[item], 'rb')
        data = pickle.load(bin_file)
        bin_file.close()

        # if item == 0:
        #     logging.info('data: {}, {}'.format(data.shape, type(data)))
        #     logging.info('data: \n{}'.format(np.round(data.flatten()[:30], 5)))
        #
        #     im = plt.imshow(data)
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        if len(self.delete_feature_idx) > 0:
            data = np.delete(data, self.delete_feature_idx, axis=-1)

        # Normalization
        norm_a = data - self.data_min
        data = np.divide(norm_a, self.norm_b)

        data = resize(data, self.shape)

        # if item == 0 and not self.bad_data:
        #     logging.info('data: {}, {}'.format(data.shape, type(data)))
        #     logging.info('data: \n{}'.format(np.round(data.flatten()[:30], 5)))
        #
        #     im = plt.imshow(data)
        #     plt.colorbar()
        #     plt.show()
        #     plt.close()

        return data

    def __getitem__(self, item):

        data = self.get_data(item)

        if self.use_triplet_loss:
            anchor = data
            # positive = add_nose_time_shift(anchor)
            positive = add_noise_salt_pepper(anchor)
            # im = plt.imshow(positive)
            # plt.colorbar()
            # plt.show()
            # plt.close()

            anchor = data.reshape(1, -1)  # flatten
            positive = positive.reshape(1, -1)  # flatten
            negative = self.get_different_data_than(item)
            negative = negative.reshape(1, -1)  # flatten
            data = np.concatenate((anchor, positive, negative), axis=0)
        else:
            data = data.reshape(1, -1)  # flatten

        source_target_data = (to_tensor(data), to_tensor(data))

        return source_target_data

    def get_different_data_than(self, item):

        while True:
            item2 = np.random.choice(len(self.data_filepaths))
            if item != item2:
                break

        data = self.get_data(item2)

        return data


# Trainers #

class EDNTrainer:
    def __init__(self, dataset, model, lr, batch_size=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = RMSELoss().to(self.device)

        self.dataset = dataset
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = len(self.dataset)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.train = False if self.dataset.bad_data else True

        if self.dataset.use_triplet_loss:
            self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())

    def train_model(self, epochs):

        cost_log = []

        if not self.train:
            logging.info('Bad data no training!')
            return cost_log

        self.model.train()
        for epoch in range(epochs):
            loss = 0
            for s_data, t_data in self.dataloader:
                s_data = s_data.to(self.device, dtype=torch.float)
                t_data = t_data.to(self.device, dtype=torch.float)

                t_data_gen, z = self.model(s_data)

                # compute training reconstruction loss
                train_loss = self.criterion(t_data_gen, t_data)

                if self.dataset.use_triplet_loss:
                    triplet_loss = self.triplet_loss(z[:, 0], z[:, 1], z[:, 2])  # anchor, positive, negative
                    train_loss = train_loss + triplet_loss

                self.optimizer.zero_grad()  # reset the gradients back to zero
                train_loss.backward()  # compute accumulated gradients
                self.optimizer.step()  # perform parameter update based on current gradients

                loss += train_loss.item()  # add the mini-batch training loss to epoch loss

            loss = loss / len(self.dataloader)  # compute the epoch training loss
            cost_log.append(loss)

            logging.info('epoch : {}/{}, loss = {:.8f}'.format(epoch + 1, epochs, loss))
            get_gpu_memory_map()

        return cost_log

    def generate_code(self, data_list):

        if not self.train:
            return torch.zeros(self.model.n_dims_code)

        self.model.eval()
        data_list = to_tensor(data_list).to(self.device, dtype=torch.float)
        with torch.no_grad():
            return self.model.encoder(data_list)

    def generate_output(self, data_list):

        if not self.train:
            return torch.zeros(self.dataset.shape)

        data_list = self.generate_code(data_list)
        with torch.no_grad():
            return self.model.decoder(data_list)

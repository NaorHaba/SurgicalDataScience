from typing import List
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd

STD_PARAMS_PATH = '/datashare/APAS/folds/std_params_fold_'


class Kinematics_Transformer():
    def __init__(self, params_path, normalization):
        params = pd.read_csv(params_path, index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]
        self.transform = self.min_max if normalization == "Min-max" \
            else self.standard if normalization == "Standard" \
            else self.tensor_float

    def tensor_float(self, features):
        return torch.tensor(features).float()

    def min_max(self, features):
        numerator = features - self.min
        denominator = self.max - self.min
        features = (numerator / denominator)
        return torch.tensor(features).float()

    def standard(self, features):
        numerator = features - self.mean
        denominator = self.std
        features = (numerator / denominator)
        return torch.tensor(features).float()


class FeatureDataset(Dataset):
    def __init__(self, surgery_folders: List, data_names: List, tasks: List, kinematics_transform=torch.tensor,
                 image_transform=torch.tensor, target_transform=torch.tensor):
        assert surgery_folders and data_names and tasks, "must receive not empty lists"
        self.surgery_folders = surgery_folders
        self.data_names = data_names
        self.tasks = tasks
        self.kinematics_transform = kinematics_transform
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        surgery_folder = self.surgery_folders[item]
        features = {}
        labels = {}
        hands_loaded = False
        for data_t in self.data_names:
            if data_t.split('.')[-1] == 'pt':
                features[data_t.split('_')[0]] = (torch.load(os.path.join(surgery_folder, data_t))).squeeze()
                if self.image_transform:
                    features[data_t.split('_')[0]] = self.image_transform(features[data_t.split('_')[0]]).float()
            else:
                features[data_t.split('.')[0]] = (np.load(os.path.join(surgery_folder, data_t))).T
                if self.kinematics_transform:
                    features[data_t.split('.')[0]] = self.kinematics_transform(features[data_t.split('.')[0]]).float()
        for task in self.tasks:
            if task=='gestures':
                original_labels = np.load(os.path.join(surgery_folder, task + '.npy'))
                original_labels = np.expand_dims(original_labels, axis=0)
                # if len(original_labels.shape) == 1 else original_labels
                labels[task] =np.array([np.array([int(original_labels[i, j][1]) for j in range(original_labels.shape[1])])
                                     for i in range(original_labels.shape[0])]).T
            else:
                if not hands_loaded:
                    original_labels = np.load(os.path.join(surgery_folder, 'tools.npy'))
                    hands = np.array([np.array([int(original_labels[i, j][1]) for j in range(original_labels.shape[1])])
                                         for i in range(original_labels.shape[0])]).T
                    hands_loaded = True
                if 'left' in task:
                    labels[task] = np.expand_dims(hands[:,0], axis=1)
                if 'right' in task:
                    labels[task] = np.expand_dims(hands[:,1], axis=1)
            # labels[task] = np.array([np.array([int(original_labels[i, j][1]) for j in range(original_labels.shape[1])])
            #                          for i in range(original_labels.shape[0])]).T
            if self.target_transform:
                labels[task] = self.target_transform(labels[task]).long()
        return features, labels

    def __len__(self):
        return len(self.surgery_folders)


def collate_inputs(batch):
    input_lengths = []
    input_masks = []
    batch_features = {}
    batch_labels = {}
    features_names = batch[0][0].keys()
    label_names = batch[0][1].keys()

    for input_name in features_names:
        batch_features[input_name] = []
        input_lengths_tmp = []
        for sample in batch:
            sample_features = sample[0][input_name]
            input_lengths_tmp.append(sample_features.shape[0])
            batch_features[input_name].append(sample_features)
        # pad
        batch_features[input_name] = torch.nn.utils.rnn.pad_sequence(batch_features[input_name], batch_first=True)
        input_lengths.append(input_lengths_tmp)
        # compute mask
        input_masks.append(batch_features[input_name] != 0)

    for input_name in label_names:
        batch_labels[input_name] = []
        input_lengths_tmp = []
        for sample in batch:
            sample_labels = sample[1][input_name]
            input_lengths_tmp.append(sample_labels.shape[0])
            batch_labels[input_name].append(sample_labels)
        # pad
        batch_labels[input_name] = torch.nn.utils.rnn.pad_sequence(batch_labels[input_name], padding_value=-100,
                                                                   batch_first=True)
        input_lengths.append(input_lengths_tmp)

    # sanity check
    assert [input_lengths[0]] * len(input_lengths) == input_lengths
    return batch_features, batch_labels, torch.tensor(input_lengths[0]), input_masks[0]

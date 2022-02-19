import pickle
from typing import List
import itertools
import torch
import numpy as np
import random
import os
import pandas as pd
import torchvision
from scipy.stats import norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
from PIL import Image
import tqdm
from model import MS_TCN, SeperateFeatureExtractor,SurgeryModel, MS_TCN_PP

STD_PARAMS_PATH = '/datashare/APAS/folds/std_params_fold_'

class Kinematics_Transformer():
    def __init__(self, params_path, normalization):
        params = pd.read_csv(params_path,index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]
        self.transform = self.min_max if normalization== "Min-max"\
            else self.standard if normalization == "Standard"\
            else self.tensor_float

    def tensor_float(self,features):
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
        for data_t in self.data_names:
            if data_t.split('.')[-1] =='pt':
                features[data_t.split('_')[0]] = (torch.load(os.path.join(surgery_folder, data_t))).squeeze()
                if self.image_transform:
                    features[data_t.split('_')[0]] = self.image_transform(features[data_t.split('_')[0]]).float()
            else:
                features[data_t.split('.')[0]] = (np.load(os.path.join(surgery_folder, data_t))).T
                if self.kinematics_transform:
                    features[data_t.split('.')[0]] = self.kinematics_transform(features[data_t.split('.')[0]]).float()
        for task in self.tasks:
            original_labels = np.load(os.path.join(surgery_folder, task + '.npy'))
            original_labels =np.expand_dims(original_labels, axis=0) if len(original_labels.shape)==1 else original_labels
            labels[task] = np.array([np.array([int(original_labels[i,j][1]) for j in range(original_labels.shape[1]) ])
                                     for i in range(original_labels.shape[0])]).T
            if self.target_transform:
                labels[task] =self.target_transform(labels[task]).long()
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
        batch_labels[input_name] = torch.nn.utils.rnn.pad_sequence(batch_labels[input_name], padding_value=-100, batch_first=True)
        input_lengths.append(input_lengths_tmp)

    # sanity check
    assert [input_lengths[0]] * len(input_lengths) == input_lengths
    return batch_features, batch_labels, torch.tensor(input_lengths[0]), input_masks[0]

#
# if __name__ == '__main__':
#
#     tasks = ['gestures']
#     folds_folder = '/datashare/apas/folds'
#     batch_size = 8
#     data_types = ['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy']
#     folds = ['fold_0','fold_1','fold_2','fold_3','fold_4']
#     data_path = '/home/student/Project/data/'
#     surgeries_per_fold = {}
#     for fold in folds:
#         fold_path = os.path.join(data_path,fold)
#         file_ptr = open(os.path.join(folds_folder, ' '.join(fold.split('_')))+'.txt', 'r')
#         fold_sur_files = [os.path.join(fold_path, x.split('.')[0]) for x in file_ptr.read().split('\n')[:-1]]
#         file_ptr.close()
#         if fold=='fold_2':
#             fold_sur_files.remove(os.path.join(os.path.join(data_path,fold),'P039_balloon2'))
#         surgeries_per_fold[fold] = fold_sur_files
#     for fold_out in folds:
#         split_num = int(fold_out.split('_')[1])
#         train_surgery_list = [surgeries_per_fold[fold] if fold != fold_out else [] for fold in folds]
#         train_surgery_list = list(itertools.chain(*train_surgery_list))
#         val_surgery_list = surgeries_per_fold[fold_out]
#         k_transform = Kinematics_Transformer(f'{STD_PARAMS_PATH}{split_num}.csv','Min-max').transform
#         ds_train = FeatureDataset(train_surgery_list, data_types, tasks, kinematics_transform=k_transform)
#         dl_train = DataLoader(ds_train, batch_size=batch_size, collate_fn=collate_inputs)
#         ds_val = FeatureDataset(val_surgery_list, data_types, tasks,kinematics_transform=k_transform)
#         dl_val = DataLoader(ds_val, batch_size=batch_size, collate_fn=collate_inputs)
#         #TODO: adjust, add to wandb, normalize kinematicks
#         mstcn = MS_TCN(3, 5, 10, 2048 * 2 + 36, 3)
#         sfe = SeperateFeatureExtractor(nn.Identity(), nn.Identity(), nn.Identity())
#         sm = SurgeryModel(sfe, mstcn)
#         for batch in dl_train:
#             batch_input, batch_target, lengths, mask = batch
#             results = sm(batch_input, lengths, mask)
#             break
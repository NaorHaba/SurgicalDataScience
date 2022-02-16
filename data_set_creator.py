import pickle
from typing import List

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
#%%
# TODO: adjust for adam's code

def create_dataset(extractor,folds_folder="/datashare/apas/folds",features_path="/datashare/apas/kinematics_npy/",frames_path = "/datashare/apas/frames/",
                   sample_rate = 6, features = ['top','side','kinematics'], labels_path ="/datashare/apas/transcriptions",
                   labels_type = ['gestures','tools'], save_path = '/home/student/Project/data'):
    for file in tqdm.tqdm(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".txt") and "fold" in filename:
            file_ptr = open(os.path.join(folds_folder, filename), 'r')
            fold_sur_files = [x.split('.')[0] for x in file_ptr.read().split('\n')[:-1]]
            file_ptr.close()
            folder_file = '_'.join(filename.split('.')[0].split(' '))
            save_dir_name = os.path.join(save_path,folder_file)
            if folder_file not in os.listdir(save_path):
                os.mkdir(save_dir_name)
            for sur in tqdm.tqdm(fold_sur_files):
                sur_dir = os.path.join(save_dir_name,sur)
                if sur not in os.listdir(save_dir_name):
                    os.mkdir(sur_dir)
                else:
                    continue
                sur_frames_path = f'{frames_path}{sur}'
                k_len, t_len, s_len = 0,0,0
                if 'top' in features:
                    t_len = int(max(os.listdir(sur_frames_path+'_top')).split('_')[1].split('.jpg')[0])
                if 'side' in features:
                    s_len =  int(max(os.listdir(sur_frames_path+'_side')).split('_')[1].split('.jpg')[0])
                if 'kinematics' in features:
                    k_array = np.load(features_path + sur + '.npy')
                    k_len = k_array.shape[1]
                min_len = min(k_len,t_len,s_len)
                samples_ind = range(1,min_len,sample_rate)
                df = pd.read_csv(f'{labels_path}_gestures/{sur}.txt', header=None, names=['start','end','label'], sep=' ')
                gestures = np.array([df[(df.start<=i)&(df.end>=i)]['label'].item() for i in samples_ind])
                np.save(f'{sur_dir}/gestures.npy',gestures)
                if 'tools' in labels_type:
                    hands_labels = np.zeros((2,len(samples_ind))).astype('str')
                    for i,hand in enumerate(['tools_left','tools_right']):
                        df = pd.read_csv(f'{labels_path}_{hand}_new/{sur}.txt', header=None,
                                         names=['start', 'end', 'label'], sep=' ')
                        cur_hand_labels =  np.array([df[(df.start<=i)&(df.end>=i)]['label'].item() for i in samples_ind])
                        hands_labels[i,:] = cur_hand_labels
                    np.save(f'{sur_dir}/tools.npy', hands_labels)
                if 'kinematics' in features:
                    k_array_sampled = k_array[:,samples_ind]
                    np.save(f'{sur_dir}/kinematics.npy', k_array_sampled)
                if 'top' in features:
                    top_frames = surgery_frames(samples_ind,sur_frames_path,'top')
                    # np.save(f'{sur_dir}/top.npy', top_frames.numpy())
                    top_frames = extractor.extractor_transform(top_frames)
                    top_frames_split = top_frames.split(125)
                    for i in range(len(top_frames_split)):
                        if i == 0:
                            top_features = extractor(top_frames_split[0])
                        else:
                            tmp_features = extractor(top_frames_split[i])
                            top_features = torch.cat((top_features, tmp_features))
                    torch.save(top_features,f'{sur_dir}/top_resnet.pt')
                if 'side' in features:
                    side_frames = surgery_frames(samples_ind,sur_frames_path,'side')
                    # np.save(f'{sur_dir}/side.npy', side_frames.numpy())
                    # torch.save(side_frames,f'{sur_dir}/side.pt')
                    side_frames = extractor.extractor_transform(side_frames)
                    side_frames_split = side_frames.split(125)
                    for i in range(len(side_frames_split)):
                        if i == 0:
                            side_features = extractor(side_frames_split[0])
                        else:
                            tmp_features = extractor(side_frames_split[i])
                            side_features = torch.cat((side_features, tmp_features))
                    torch.save(side_features,f'{sur_dir}/side_resnet.pt')



def surgery_frames(samples_ind, surgery_path, video_type):
    path = f'{surgery_path}_{video_type}'
    template= '00000'
    t_list = []
    for i,frame_num in tqdm.tqdm(enumerate(samples_ind)):
        img_path = path+'/img_'+template[:-len(str(frame_num))]+str(frame_num)+'.jpg'
        img = Image.open(img_path)
        transformer = transforms.Compose(
            [transforms.ToTensor()])
        t_list.append(transformer(img).requires_grad_(False))
    return torch.stack(t_list)

class Resnet_feature_extractor(nn.Module):
    """
    This class is the ResNet-50 based class used for feature extraction used in the advanced part of the project.
    The class enables extraction and storing of chosen intermediate results of input images from a forward pass in the
    ResNet-50 model.
    """
    def __init__(self, device):
        super(Resnet_feature_extractor, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True).to(device)
        self.device = device
        self.transformer = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.Resize(
            size=224), transforms.CenterCrop((224, 224))])
        # freeze all Renset parameters since we’re only optimizing the target image
        for param in self.resnet_model.parameters():
            param.requires_grad_(False)

    def extractor_transform (self, x):
        return self.transformer(x)

    def forward(self, x):
        x = x.to(self.device)
        for (layer_name, layer) in self.resnet_model.named_children():
            if layer_name not in ['fc']:
                x = layer(x)
        return x
#%%
class FeatureDataset(Dataset):
    def __init__(self, surgery_folders: List, data_types: List, tasks: List,
                 transform=torch.tensor, target_transform=torch.tensor):
        assert surgery_folders and data_types and tasks, "must receive not empty lists"
        self.surgery_folders = surgery_folders
        self.data_types = data_types
        self.tasks = tasks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        surgery_folder = self.surgery_folders[item]
        features = {}
        labels = {}
        for data_t in self.data_types:
            if data_t.split('.')[-1] =='pt':
                features[data_t.split('_')[0]] = (torch.load(os.path.join(surgery_folder, data_t))).squeeze()
                if self.transform:
                    features[data_t.split('_')[0]] = self.transform(features[data_t.split('_')[0]])
            else:
                features[data_t.split('.')[0]] = (np.load(os.path.join(surgery_folder, data_t))).T
                if self.transform:
                    features[data_t.split('.')[0]] = self.transform(features[data_t.split('.')[0]])
        for task in self.tasks:
            original_labels = np.load(os.path.join(surgery_folder, task + '.npy'))
            original_labels =np.expand_dims(original_labels, axis=0) if len(original_labels.shape)==1 else original_labels
            labels[task] = np.array([np.array([int(original_labels[i,j][1]) for j in range(original_labels.shape[1]) ])
                                     for i in range(original_labels.shape[0])]).T
            if self.target_transform:
                labels[task] =self.target_transform(labels[task])
        return features, labels

    def __len__(self):
        return len(self.surgery_folders)
#%%
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
    return batch_features, batch_labels, input_lengths[0], input_masks[0]

#%%
if __name__ == '__main__':
    # extractor = Resnet_feature_extractor(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # folds_folder = "/datashare/apas/folds"
    # features_path = "/datashare/apas/kinematics_npy/"
    # frames_path = "/datashare/apas/frames/"
    # sample_rate = 6
    # features = ['top', 'side', 'kinematics']
    # labels_path = "/datashare/apas/transcriptions"
    # labels_type = ['gestures', 'tools']
    # save_path = '/home/student/Project/data'
    # print('creating data set')
    # create_dataset(extractor = extractor, folds_folder=folds_folder, features_path=features_path,
    #                    frames_path=frames_path, sample_rate=sample_rate, features=features,
    #                    labels_path=labels_path, labels_type=labels_type, save_path=save_path)
    surgery_folders_list = ['/home/student/Project/data/fold_0/P025_balloon2',
                            '/home/student/Project/data/fold_0/P025_tissue2',
                            '/home/student/Project/data/fold_0/P025_balloon1',
                            '/home/student/Project/data/fold_0/P025_tissue1',
                            '/home/student/Project/data/fold_0/P020_tissue2',
                            '/home/student/Project/data/fold_0/P020_balloon1',
                            '/home/student/Project/data/fold_0/P020_tissue1',
                            '/home/student/Project/data/fold_0/P034_balloon1',
                            '/home/student/Project/data/fold_0/P034_balloon2',
                            '/home/student/Project/data/fold_0/P034_tissue1',
                            '/home/student/Project/data/fold_0/P034_tissue2',
                            '/home/student/Project/data/fold_0/P016_tissue2',
                            '/home/student/Project/data/fold_0/P016_balloon2',
                            '/home/student/Project/data/fold_0/P016_balloon1',
                            '/home/student/Project/data/fold_0/P016_tissue1',
                            '/home/student/Project/data/fold_0/P022_tissue1',
                            '/home/student/Project/data/fold_0/P022_balloon1',
                            '/home/student/Project/data/fold_0/P022_balloon2',
                            '/home/student/Project/data/fold_0/P022_tissue2',
                            ]
    tasks = ['tools','gestures']
    data_types = ['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy']
    ds = FeatureDataset(surgery_folders_list, data_types, tasks)
    dl = DataLoader(ds, batch_size=8, collate_fn=collate_inputs)
    for batch in dl:
        tmp = batch
        break

# In this file we define the functions for the offline feature extraction of the videos. The results are save on the machine.

import torch
import numpy as np
import os
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
from PIL import Image
import tqdm


def extract_features(folds_folder, features_path, frames_path, sample_rate, features, labels_path, labels_type,
                     save_path='/home/student/Project/data'):
    extractor = Resnet_feature_extractor(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    create_dataset(extractor=extractor, folds_folder=folds_folder, features_path=features_path,
                   frames_path=frames_path, sample_rate=sample_rate, features=features,
                   labels_path=labels_path, labels_type=labels_type, save_path=save_path)


def create_dataset(extractor, folds_folder="/datashare/apas/folds", features_path="/datashare/apas/kinematics_npy/",
                   frames_path="/datashare/apas/frames/",
                   sample_rate=6, features=['top', 'side', 'kinematics'], labels_path="/datashare/apas/transcriptions",
                   labels_type=['gestures', 'tools'], save_path='/home/student/Project/data'):
    for file in tqdm.tqdm(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".txt") and "fold" in filename:
            file_ptr = open(os.path.join(folds_folder, filename), 'r')
            fold_sur_files = [x.split('.')[0] for x in file_ptr.read().split('\n')[:-1]]
            file_ptr.close()
            folder_file = '_'.join(filename.split('.')[0].split(' '))
            folder_file_aug = '_'.join(filename.split('.')[0].split(' ')) + '_augmentation'
            save_dir_name = os.path.join(save_path, folder_file)
            save_dir_name_aug = os.path.join(save_path, folder_file_aug)
            if folder_file not in os.listdir(save_path):
                os.mkdir(save_dir_name)
            if folder_file_aug not in os.listdir(save_path):
                os.mkdir(save_dir_name_aug)
            for sur in tqdm.tqdm(fold_sur_files):
                sur_dir = os.path.join(save_dir_name, sur)
                sur_dir_aug = os.path.join(save_dir_name_aug, sur)
                if sur not in os.listdir(save_dir_name):
                    os.mkdir(sur_dir)
                if sur not in os.listdir(save_dir_name_aug):
                    os.mkdir(sur_dir_aug)
                if sur == 'P039_balloon2':  # ignored because of problems with annotation
                    continue
                print(sur)
                sur_frames_path = f'{frames_path}{sur}'
                k_len, t_len, s_len = 0, 0, 0
                if 'top' in features:
                    t_len = int(max(os.listdir(sur_frames_path + '_top')).split('_')[1].split('.jpg')[0])
                if 'side' in features:
                    s_len = int(max(os.listdir(sur_frames_path + '_side')).split('_')[1].split('.jpg')[0])
                if 'kinematics' in features:
                    k_array = np.load(features_path + sur + '.npy')
                    k_len = k_array.shape[1]
                min_len = min(k_len, t_len,
                              s_len) if sur != 'P020_balloon2' else 7405  # to avoid an error with P020_baloon.
                samples_ind = range(1, min_len, sample_rate)
                df = pd.read_csv(f'{labels_path}_gestures/{sur}.txt', header=None, names=['start', 'end', 'label'],
                                 sep=' ')
                gestures = np.array([df[(df.start <= i) & (df.end >= i)]['label'].item() for i in samples_ind])
                np.save(f'{sur_dir}/gestures.npy', gestures)
                if 'tools' in labels_type:
                    hands_labels = np.zeros((2, len(samples_ind))).astype('str')
                    for i, hand in enumerate(['tools_left', 'tools_right']):
                        df = pd.read_csv(f'{labels_path}_{hand}_new/{sur}.txt', header=None,
                                         names=['start', 'end', 'label'], sep=' ')
                        cur_hand_labels = np.array(
                            [df[(df.start <= i) & (df.end >= i)]['label'].item() for i in samples_ind])
                        hands_labels[i, :] = cur_hand_labels
                    np.save(f'{sur_dir}/tools.npy', hands_labels)
                if 'kinematics' in features:
                    k_array_sampled = k_array[:, samples_ind]
                    np.save(f'{sur_dir}/kinematics.npy', k_array_sampled)
                if 'top' in features:
                    top_frames = surgery_frames(samples_ind, sur_frames_path, 'top')
                    for dir, augment in [(sur_dir, False), (sur_dir_aug, True)]:
                        top_frames_transformed = extractor.extractor_transform(top_frames, augment=augment)
                        top_frames_split = top_frames_transformed.split(125)
                        for i in range(len(top_frames_split)):
                            if i == 0:
                                top_features = extractor(top_frames_split[0])
                            else:
                                tmp_features = extractor(top_frames_split[i])
                                top_features = torch.cat((top_features, tmp_features))
                        torch.save(top_features, f'{dir}/top_resnet.pt')
                if 'side' in features:
                    side_frames = surgery_frames(samples_ind, sur_frames_path, 'side')
                    for dir, augment in [(sur_dir, False), (sur_dir_aug, True)]:
                        side_frames_transformed = extractor.extractor_transform(side_frames, augment=augment)
                        side_frames_split = side_frames_transformed.split(125)
                        for i in range(len(side_frames_split)):
                            if i == 0:
                                side_features = extractor(side_frames_split[0])
                            else:
                                tmp_features = extractor(side_frames_split[i])
                                side_features = torch.cat((side_features, tmp_features))
                        torch.save(side_features, f'{dir}/side_resnet.pt')


def surgery_frames(samples_ind, surgery_path, video_type):
    path = f'{surgery_path}_{video_type}'
    template = '00000'
    t_list = []
    transformer = transforms.Compose([transforms.ToTensor()])
    for i, frame_num in tqdm.tqdm(enumerate(samples_ind)):
        img_path = path + '/img_' + template[:-len(str(frame_num))] + str(frame_num) + '.jpg'
        img = Image.open(img_path)
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
        self.transformations = [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), transforms.Resize(
            size=224), transforms.CenterCrop((224, 224))]
        # freeze all Renset parameters since we’re only optimizing the target image
        for param in self.resnet_model.parameters():
            param.requires_grad_(False)

    def extractor_transform(self, x, augment=False):
        if augment:
            transformations = self.transformations.copy() + [transforms.functional.hflip]
        else:
            transformations = self.transformations.copy()
        transformer = transforms.Compose(transformations)
        return transformer(x)

    def forward(self, x):
        x = x.to(self.device)
        for (layer_name, layer) in self.resnet_model.named_children():
            if layer_name not in ['fc']:
                x = layer(x)
        return x


if __name__ == '__main__':
    extractor = Resnet_feature_extractor(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    folds_folder = "/datashare/apas/folds"
    features_path = "/datashare/apas/kinematics_npy/"
    frames_path = "/datashare/apas/frames/"
    sample_rate = 6
    features = ['top', 'side', 'kinematics']
    labels_path = "/datashare/apas/transcriptions"
    labels_type = ['gestures', 'tools']
    save_path = '/home/student/Project/data'
    print('creating data set')
    create_dataset(extractor=extractor, folds_folder=folds_folder, features_path=features_path,
                   frames_path=frames_path, sample_rate=sample_rate, features=features,
                   labels_path=labels_path, labels_type=labels_type, save_path=save_path)

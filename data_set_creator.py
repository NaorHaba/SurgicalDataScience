import torch
import numpy as np
import random
import os
import pandas as pd
from scipy.stats import norm
from torchvision import transforms, models
import torch.nn as nn
#%%
list_of_train_examples = []
folds_folder = "/datashare/apas/folds"
features_path =  "/datashare/apas/kinematics_npy/"
frames_path = "/datashare/apas/frames/"
sample_rate = 6

# TODO: create min len function
# TODO: adjust for adam's code

def create_dataset(folds_folder="/datashare/apas/folds",features_path="/datashare/apas/kinematics_npy/",frames_path = "/datashare/apas/frames/",
                   sample_rate = 6, features = ['top','side','kinematics']):
    for file in os.listdir(folds_folder):
        filename = os.fsdecode(file)
        print(filename)
        if filename.endswith(".txt") and "fold" in filename:
            file_ptr = open(os.path.join(folds_folder, filename), 'r')
            fold_sur_files = [x.split('.')[0] for x in file_ptr.read().split('\n')[:-1]]
            file_ptr.close()
            k_len, t_len, s_len = 0,0,0
            for sur in fold_sur_files:
                sur_frames_path = f'{frames_path}/{sur}'
                if 'top' in features:
                    t_len = int(max(os.listdir(sur_frames_path+'_top')).split('_')[1].split('.jpg')[0])
                if 'side' in features:
                    s_len =  int(max(os.listdir(sur_frames_path+'_side')).split('_')[1].split('.jpg')[0])
                if 'kinematics' in features:
                    k_array = np.load(features_path + sur + '.npy')
                    k_len = k_array.shape[1]
                min_len = min(k_len,t_len,s_len)
                samples_ind = range(0,min_len,sample_rate)
                if 'kinematics' in features:
                    k_array_sampled = k_array[:,samples_ind]
                print(samples_ind)
                break
            # with open(f'{filename.replace(".txt","")}_paths.txt','a') as file:



def image_feature_extraction(samples_ind, surgery_path, video_type):


#%%
class Resnet_feature_extractor(nn.Module):
    """
    This class is the ResNet-50 based class used for feature extraction used in the advanced part of the project.
    The class enables extraction and storing of chosen intermediate results of input images from a forward pass in the
    ResNet-50 model.
    """
    def __init__(self, device, blocks_to_extract):
        super(Resnet_feature_extractor, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True).to(device)
        # freeze all Renset parameters since we’re only optimizing the target image
        for param in self.resnet_model.parameters():
            param.requires_grad_(False)

    def forward(self, x, img_type: str):
        for (layer_name, layer) in self.resnet_model.named_children():
            if layer_name not in ['avgpool', 'fc']:
                x = layer(x)
        return x
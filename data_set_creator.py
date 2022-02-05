import torch
import numpy as np
import random
import os
import pandas as pd
import torchvision
from scipy.stats import norm
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
from PIL import Image
import tqdm
#%%

# TODO: create min len function
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
            save_dir_name = os.path.join(save_path,'_'.join(filename.split('.')[0].split(' ')))
            os.mkdir(save_dir_name)
            for sur in tqdm.tqdm(fold_sur_files):
                sur_dir = os.path.join(save_dir_name,sur)
                os.mkdir(sur_dir)
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
                    hands_labels = np.zeros((2,len(samples_ind)))
                    for i,hand in enumerate(['tools_left','tools_right']):
                        df = pd.read_csv(f'{labels_path}_{hand}/{sur}.txt', header=None,
                                         names=['start', 'end', 'label'], sep=' ')
                        hands_labels[i,:] = np.array([df[(df.start<=i)&(df.end>=i)]['label'].item() for i in samples_ind])
                    np.save(f'{sur_dir}/tools.npy', hands_labels)
                if 'kinematics' in features:
                    k_array_sampled = k_array[:,samples_ind]
                    np.save(f'{sur_dir}/kinematics.npy', k_array_sampled)
                if 'top' in features:
                    features = image_feature_extraction_s(samples_ind,sur_frames_path,'top',extractor)
                    np.save(f'{sur_dir}/top.npy', features)
                if 'side' in features:
                    features = image_feature_extraction_s(samples_ind,sur_frames_path,'side',extractor)
                    np.save(f'{sur_dir}/side.npy', features)


def image_feature_extraction_s(samples_ind, surgery_path, video_type,extractor):
    path = f'{surgery_path}_{video_type}'
    template= '00000'
    t_list = []
    for i,frame_num in tqdm.tqdm(enumerate(samples_ind)):
        img_path = path+'/img_'+template[:-len(str(frame_num))]+str(frame_num)+'.jpg'
        img = Image.open(img_path)
        transformer = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),transforms.Resize(size=224), transforms.CenterCrop((224,224))])
        t_list.append(transformer(img).requires_grad_(False))
    imgs= torch.stack(t_list)
    features = extractor(imgs)
    return features.squeeze()


#%%
class Resnet_feature_extractor(nn.Module):
    """
    This class is the ResNet-50 based class used for feature extraction used in the advanced part of the project.
    The class enables extraction and storing of chosen intermediate results of input images from a forward pass in the
    ResNet-50 model.
    """
    def __init__(self, device):
        super(Resnet_feature_extractor, self).__init__()
        self.resnet_model = models.resnet50(pretrained=True).to(device)
        # freeze all Renset parameters since we’re only optimizing the target image
        for param in self.resnet_model.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        for (layer_name, layer) in self.resnet_model.named_children():
            if layer_name not in ['fc']:
                x = layer(x)
        print(x.shape)
        return x
#%%


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
    create_dataset(extractor = extractor, folds_folder=folds_folder, features_path=features_path,
                       frames_path=frames_path, sample_rate=sample_rate, features=features,
                       labels_path=labels_path, labels_type=labels_type, save_path=save_path)
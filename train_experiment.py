# Created by Adam Goldbraikh - Scalpel Lab Technion (Edited by Naor Haba and Lior Yariv :) )
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from Trainer import Trainer
import os
import argparse
from datetime import datetime
from termcolor import colored
import random
import logging
import itertools
from datasets import FeatureDataset, collate_inputs, KinematicsTransformer
from model import MS_TCN, MS_TCN_PP, SeperateFeatureExtractor, SurgeryModel, MT_RNN_dp

logger = logging.getLogger(__name__)
EXTRACTED_DATA_PATH = '/home/student/Project/data/'
FOLDS_FOLDER_PATH = '/datashare/APAS/folds'
STD_PARAMS_PATH = '/datashare/APAS/folds/std_params_fold_'
GESTURES_NUM_CLASSES = 6
TOOLS_NUM_CLASSES = 4
SAMPLE_RATE = 6
ACTIVATIONS = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'tanh': nn.Tanh}


def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_name', default="LSTMS")

    parser.add_argument('--dataset', choices=['APAS'], default="APAS")
    parser.add_argument('--data_types', choices=['top', 'side', 'kinematics'], nargs='+',
                        default=['kinematics'])
    # parser.add_argument('--data_names', choices=['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy'], nargs='+',
    #                     default=['top_resnet.pt', 'side_resnet.pt'])
    parser.add_argument('--data_names', choices=['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy'], nargs='+',
                        default=['kinematics.npy'])

    parser.add_argument('--task_str', default='gestures, tools_left, tools_right')

    parser.add_argument('--test_split', choices=[0, 1, 2, 3, 4], default=None)

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="MSTCN_runs_fixed", type=str)
    parser.add_argument('--entity', default="surgical_data_science", type=str)
    parser.add_argument('--group', default=dt_string + " group ", type=str)
    parser.add_argument('--use_gpu_num', default="0", type=str)

    parser.add_argument('--time_series_model', choices=['MSTCN', 'MSTCN++', 'LSTM', 'GRU'], default='GRU', type=str)
    parser.add_argument('--feature_extractor', choices=['separate', 'linear_kinematics'], default='separate', type=str)
    parser.add_argument('--augmentation', default=True, type=bool)
    parser.add_argument('--flip_hands', default=True, type=bool)

    parser.add_argument('--num_stages', default=5, type=int)
    parser.add_argument('--num_layers', default=9, type=int)
    parser.add_argument('--num_f_maps', default=1024, type=int)
    parser.add_argument('--activation', choices=['tanh'], default='tanh', type=str)
    parser.add_argument('--dropout', default=0.10402892383683587, type=float)
    parser.add_argument('--num_layers_rnn', default=4, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--dropout_rnn', default=0.4, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--normalization', choices=['Standard'], default='Standard', type=str)
    parser.add_argument('--lr', default=0.00876569212062032, type=float)

    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--loss_factor', default=0.3, type=float)
    parser.add_argument('--T', default=9, type=int)
    parser.add_argument('--hands_factor', default=1, type=int)

    args = parser.parse_args()

    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=1538574472):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# use the full temporal resolution @ 30Hz
def splits_dict(data_path, folds_folder):
    surgeries_per_fold = {}
    vids_per_fold = {}
    surgeries_augmented_per_fold = {}
    for split in range(5):
        fold_path = os.path.join(data_path, 'fold_' + str(split))
        file_ptr = open(os.path.join(folds_folder, f'fold {split}.txt'), 'r')
        vids_list = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        if split == 2:  ## problems with annotations. Adam said we can ignore
            vids_list.remove('P039_balloon2.csv')
        fold_sur_files = [os.path.join(fold_path, x.split('.')[0]) for x in vids_list]
        surgeries_per_fold[split] = fold_sur_files
        fold_sur_augmented_files = [os.path.join(fold_path + '_augmentation', x.split('.')[0]) for x in vids_list]
        surgeries_augmented_per_fold[split] = fold_sur_augmented_files
        vids_per_fold[split] = vids_list
    return surgeries_per_fold, vids_per_fold, surgeries_augmented_per_fold


def create_model(args):
    if args.feature_extractor == 'separate':
        fe_params = {d + '_fe': nn.Identity() for d in args.data_types}
        fe = SeperateFeatureExtractor(**fe_params)
        feature_sizes = {'top': 2048, 'side': 2048, 'kinematics': 36}
    elif args.feature_extractor == 'linear_kinematics':
        fe_params = {d + '_fe': nn.Identity() for d in args.data_types if d != 'kinematics'}
        fe_params['kinematics_fe'] = nn.Sequential(nn.Linear(36, 256), nn.LeakyReLU())
        fe = SeperateFeatureExtractor(**fe_params)
        feature_sizes = {'top': 2048, 'side': 2048, 'kinematics': 256}
    else:
        raise NotImplementedError
    dims = sum([feature_sizes[d] for d in args.data_types])
    print(dims)
    activation = ACTIVATIONS[args.activation]
    if args.time_series_model == 'MSTCN':
        ts = MS_TCN(num_stages=args.num_stages, num_layers=args.num_layers, num_f_maps=args.num_f_maps, dim=dims,
                    num_classes=args.num_classes_list, activation=activation, dropout=args.dropout)
    elif args.time_series_model == 'MSTCN++':
        ts = MS_TCN_PP(num_stages=args.num_stages, num_layers=args.num_layers, num_f_maps=args.num_f_maps, dim=dims,
                       num_classes=args.num_classes_list, activation=activation, dropout=args.dropout)
    elif args.time_series_model == 'LSTM':
        ts = MT_RNN_dp(rnn_type='LSTM', input_dim=dims, num_classes_list=args.num_classes_list,
                       hidden_dim=args.hidden_dim, bidirectional=args.bidirectional,
                       dropout=args.dropout_rnn, num_layers=args.num_layers_rnn)
    else:
        ts = MT_RNN_dp(rnn_type='GRU', input_dim=dims, num_classes_list=args.num_classes_list,
                       hidden_dim=args.hidden_dim, bidirectional=args.bidirectional,
                       dropout=args.dropout_rnn, num_layers=args.num_layers_rnn)
    sm = SurgeryModel(fe, ts)
    return sm


def eval_dict_func(args, device):
    folds_folder = "/datashare/" + args.dataset + "/folds"
    features_path = "/datashare/" + args.dataset + "/kinematics_npy/"
    gt_path_gestures = "/datashare/" + args.dataset + "/transcriptions_gestures/"
    gt_path_tools_left = "/datashare/" + args.dataset + "/transcriptions_tools_left/"
    gt_path_tools_right = "/datashare/" + args.dataset + "/transcriptions_tools_right/"
    mapping_gestures_file = "/datashare/" + args.dataset + "/mapping_gestures.txt"
    mapping_tool_file = "/datashare/" + args.dataset + "/mapping_tools.txt"
    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools = 0
    actions_dict_tools = dict()
    if args.dataset == "APAS":
        file_ptr = open(mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            actions_dict_tools[a.split()[1]] = int(a.split()[0])
    eval_dict = {"features_path": features_path, "actions_dict_gestures": actions_dict_gestures,
                 "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": SAMPLE_RATE,
                 "eval_rate": args.eval_rate,
                 "gt_path_gestures": gt_path_gestures, "gt_path_tools_left": gt_path_tools_left,
                 "gt_path_tools_right": gt_path_tools_right, "task": args.task_str}
    return eval_dict


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        'WANDB_SWEEP_PARAM_PATH'
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def main(trial):
    args = parsing()
    task_factor = [1] + 2 * [args.hands_factor]
    # data_types_str = 'top,side'
    # args.data_types = data_types_str.split(',')
    args.data_names = [f'{x}_resnet.pt' if x != 'kinematics' else f'{x}.npy' for x in args.data_types]
    print(args.data_names)
    set_seed()
    logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    list_of_splits = [args.test_split] if args.test_split is not None else list(range(5))
    print(list_of_splits)
    experiment_name = args.group + " task:" + args.task_str
    args.group = experiment_name
    logger.info(colored(experiment_name, "green"))
    surgeries_per_fold, vids_per_fold, surgeries_augmented_per_fold = splits_dict(EXTRACTED_DATA_PATH,
                                                                                  FOLDS_FOLDER_PATH)
    num_classes_list = []
    tasks = args.task_str.split(', ')
    for task in tasks:
        num_classes_list += [GESTURES_NUM_CLASSES] if task == 'gestures' else [TOOLS_NUM_CLASSES]
    args.num_classes_list = num_classes_list
    accs = []
    for split_num in list_of_splits:
        # args.test_split = split_num
        logger.info("working on split number: " + str(split_num))
        reset_wandb_env()
        train_surgery_list = [surgeries_per_fold[fold] if fold != split_num else [] for fold in surgeries_per_fold]
        if args.augmentation and ('top' in args.data_types or 'side' in args.data_types):
            train_surgery_list += [surgeries_augmented_per_fold[fold] if fold != split_num else [] for fold in
                                   surgeries_augmented_per_fold]
        else:
            args.augmentation = False
        train_surgery_list = list(itertools.chain(*train_surgery_list))
        val_surgery_list = surgeries_per_fold[split_num]
        k_transform = KinematicsTransformer(f'{STD_PARAMS_PATH}{split_num}.csv', args.normalization).transform
        ds_train = FeatureDataset(train_surgery_list, args.data_names, tasks, k_transform, flip_hands=args.flip_hands)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs, shuffle=True)
        ds_val = FeatureDataset(val_surgery_list, args.data_names, tasks, k_transform, flip_hands=args.flip_hands)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs)
        model = create_model(args)
        trainer = Trainer(num_classes=args.num_classes_list, model=model, task=tasks, device=device)
        eval_dict = eval_dict_func(args, device)
        eval_results, train_results, best_results = trainer.train(dl_train, dl_val, num_epochs=args.num_epochs,
                                                                  learning_rate=args.lr,
                                                                  eval_dict=eval_dict,
                                                                  list_of_vids=vids_per_fold[split_num],
                                                                  args=args, test_split=split_num,
                                                                  loss_factor=args.loss_factor, T=args.T,
                                                                  task_factor=task_factor)
        accs.append(best_results['Acc gesture'])
    return np.mean(accs)


if __name__ == '__main__':
    accs = main('trial')

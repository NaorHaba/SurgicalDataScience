# Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader

from Trainer import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored, cprint
import random
import logging
import itertools
from data_set_creator import FeatureDataset, collate_inputs, Kinematics_Transformer
from model import MS_TCN, MS_TCN_PP, SeperateFeatureExtractor, SurgeryModel

logger = logging.getLogger(__name__)
EXTRACTED_DATA_PATH = '/home/student/Project/data/'
FOLDS_FOLDER_PATH = '/datashare/APAS/folds'
STD_PARAMS_PATH = '/datashare/APAS/folds/std_params_fold_'
GESTURES_NUM_CLASSES = 6
# TOOLS_NUM_CLASSES = 4
SAMPLE_RATE = 6
ACTIVATIONS = {'relu': nn.ReLU, 'lrelu': nn.LeakyReLU, 'tanh': nn.Tanh}


def parsing():
    dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['APAS'], default="APAS")
    parser.add_argument('--data_types', choices=['top', 'side', 'kinematics'], nargs='+',
                        default=['top', 'side', 'kinematics'])
    parser.add_argument('--data_names', choices=['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy'], nargs='+',
                        default=['top_resnet.pt', 'side_resnet.pt', 'kinematics.npy'])
    parser.add_argument('--task', choices=['tools', 'gestures'], nargs='+', default=['gestures'])
    parser.add_argument('--test_split', choices=[0, 1, 2, 3, 4], default=None)

    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', type=str)
    parser.add_argument('--project', default="MSTCN", type=str)
    parser.add_argument('--entity', default="surgical_data_science", type=str)
    parser.add_argument('--group', default=dt_string + " group ", type=str)
    parser.add_argument('--use_gpu_num', default="0", type=str)

    parser.add_argument('--time_series_model', choices=['MSTCN', 'MSTCN++'], default='MSTCN', type=str)
    parser.add_argument('--feature_extractor', choices=['separate'], default='separate', type=str)

    parser.add_argument('--num_stages', default=3, type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--num_f_maps', default=10, type=int)
    parser.add_argument('--activation', choices=['relu', 'lrelu', 'tanh'], default='relu', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)

    parser.add_argument('--eval_rate', default=1, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--normalization', choices=['none', 'Min-max', 'Standard'], default='Min-max', type=str)
    parser.add_argument('--lr', default=0.03, type=float)
    parser.add_argument('--num_epochs', default=1, type=int)

    args = parser.parse_args()

    assert 0 <= args.dropout <= 1
    return args


def set_seed(seed=1538574472):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# use the full temporal resolution @ 30Hz

def splits_dict(list_of_splits, data_path, folds_folder):
    surgeries_per_fold = {}
    vids_per_fold = {}
    for split in list_of_splits:
        fold_path = os.path.join(data_path, 'fold_' + str(split))
        file_ptr = open(os.path.join(folds_folder, f'fold {split}.txt'), 'r')
        vids_list = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        if split == 2:
            vids_list.remove('P039_balloon2.csv')
        fold_sur_files = [os.path.join(fold_path, x.split('.')[0]) for x in vids_list]
        surgeries_per_fold[split] = fold_sur_files
        vids_per_fold[split] = vids_list
    return surgeries_per_fold, vids_per_fold


def create_model(args):
    if args.feature_extractor == 'separate':
        fe_params = {d + '_fe': nn.Identity() for d in args.data_types}
        fe = SeperateFeatureExtractor(**fe_params)
    else:
        raise ValueError
    # for now we assume nn.identity() is always passed as feature extractor
    feature_sizes = {'top': 2048, 'side': 2048, 'kinematics': 36}
    dims = sum([feature_sizes[d] for d in args.data_types])
    if args.time_series_model == 'MSTCN':
        ts = MS_TCN
    else:
        ts = MS_TCN_PP
    activation = ACTIVATIONS[args.activation]
    # num_stages, num_layers, num_f_maps, dim, num_classes, activation=nn.ReLU, dropout=0.1
    ts = ts(num_stages=args.num_stages, num_layers=args.num_layers, num_f_maps=args.num_f_maps, dim=dims,
            num_classes=GESTURES_NUM_CLASSES, activation=activation, dropout=args.dropout)
    sm = SurgeryModel(fe, ts)
    return sm


def eval_dict_func(args):
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
        num_classes_tools = len(actions_dict_tools)
    num_classes_gestures = len(actions_dict_gestures)
    num_classes_list = [num_classes_gestures]
    eval_dict = {"features_path": features_path, "actions_dict_gestures": actions_dict_gestures,
                 "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": SAMPLE_RATE,
                 "eval_rate": args.eval_rate,
                 "gt_path_gestures": gt_path_gestures, "gt_path_tools_left": gt_path_tools_left,
                 "gt_path_tools_right": gt_path_tools_right, "task": args.task}
    return eval_dict

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


if __name__ == '__main__':
    args = parsing()
    # sample_rate = 6  # downsample the frequency to 5Hz - the data files created in feature_extractor use sample rate=6
    set_seed()
    logger.info(args)  # TODO : what is this?
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    list_of_splits = args.test_split if args.test_split else list(range(5))
    experiment_name = args.group + " task:" + ' '.join(args.task)
    args.group = experiment_name
    logger.info(colored(experiment_name, "green"))
    # summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
    # if not args.debugging:
    #     if not os.path.exists(summaries_dir):
    #         os.makedirs(summaries_dir)
    # full_eval_results = pd.DataFrame()
    # full_train_results = pd.DataFrame()
    surgeries_per_fold, vids_per_fold = splits_dict(list_of_splits, EXTRACTED_DATA_PATH, FOLDS_FOLDER_PATH)
    for split_num in list_of_splits:
        args.test_split = split_num
        logger.info("working on split number: " + str(split_num))
        reset_wandb_env()
        # model_dir = "./models/" + args.dataset + "/" + experiment_name + "/split_" + split_num
        # if not args.debugging:
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir)
        train_surgery_list = [surgeries_per_fold[fold] if fold != split_num else [] for fold in surgeries_per_fold]
        train_surgery_list = list(itertools.chain(*train_surgery_list))
        val_surgery_list = surgeries_per_fold[split_num]
        k_transform = Kinematics_Transformer(f'{STD_PARAMS_PATH}{split_num}.csv', args.normalization).transform
        ds_train = FeatureDataset(train_surgery_list, args.data_names, args.task, k_transform)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, collate_fn=collate_inputs)
        ds_val = FeatureDataset(val_surgery_list, args.data_names, args.task, k_transform)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, collate_fn=collate_inputs)
        # TODO
        model = create_model(args)
        trainer = Trainer(num_classes=GESTURES_NUM_CLASSES, model=model, task=args.task, device=device)
        eval_dict = eval_dict_func(args)
        eval_results, train_results = trainer.train(dl_train, dl_val, num_epochs=args.num_epochs, learning_rate=args.lr,
                                                    eval_dict=eval_dict, list_of_vids=vids_per_fold[split_num],
                                                    args=args)
        #
        # if not args.debugging:
        #     eval_results = pd.DataFrame(eval_results)
        #     train_results = pd.DataFrame(train_results)
        #     eval_results = eval_results.add_prefix('split_' + str(split_num) + '_')
        #     train_results = train_results.add_prefix('split_' + str(split_num) + '_')
        #     full_eval_results = pd.concat([full_eval_results, eval_results], axis=1)
        #     full_train_results = pd.concat([full_train_results, train_results], axis=1)
        #     full_eval_results.to_csv(summaries_dir + "/evaluation_results.csv", index=False)
        #     full_train_results.to_csv(summaries_dir + "/train_results.csv", index=False)

# Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

from model import *
import sys
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import *
import wandb
from datetime import datetime
import tqdm


class Trainer:
    def __init__(self, dim, num_classes_list, hidden_dim=64, dropout=0.4, num_layers=3, offline_mode=True,
                 task="gestures", device="cuda",
                 network='LSTM', debagging=False):

        self.model = MT_RNN_dp(network, input_dim=dim, hidden_dim=hidden_dim, num_classes_list=num_classes_list,
                               bidirectional=offline_mode, dropout=dropout, num_layers=num_layers)

        self.debugging = debagging
        self.network = network
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes_list = num_classes_list
        self.task = task

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(project=args.project, group=args.group,
                       name="split: " + args.split, entity=args.entity,  # FIXME: we added entity
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0

            while batch_gen.has_next():
                batch_input, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target_gestures, mask = batch_input.to(self.device), batch_target_gestures.to(
                    self.device), mask.to(self.device)

                optimizer.zero_grad()
                lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                predictions1 = self.model(batch_input, lengths)
                predictions1 = (predictions1[0] * mask).unsqueeze_(0)

                loss = 0
                for p in predictions1:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]),
                                    batch_target_gestures.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted1 = torch.max(predictions1[-1].data, 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][
                                                               :lengths[i]]).float().sum().item()
                    total1 += lengths[i]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            if not self.debugging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(
                                                                                                          correct1) / total1))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct1) / total1}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)

        return eval_results_list, train_results_list

    def evaluate(self, eval_dict, batch_gen):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            recognition1_list = []

            for seq in list_of_vids:
                # print vid
                features = np.load(features_path + seq.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions1 = self.model(input_x, torch.tensor([features.shape[1]]))
                predictions1 = predictions1[0].unsqueeze_(0)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()

                recognition1 = []
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                                                                      list(actions_dict_gesures.values()).index(
                                                                          predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            results.update(results1)

            self.model.train()
            return results

# Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    def __init__(self, num_classes, model, task=["gestures"], device="cuda"):

        self.model = model
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes_list = num_classes
        self.task = task

    def train(self, train_data_loader, test_data_loader, num_epochs, learning_rate, eval_dict, list_of_vids, args,
              test_split,early_stop=8, loss_factor=0.15, T=16):
        # ** batch_gen changed to train_data_loader and test_data_loader

        # ** old -
        # number_of_seqs = len(batch_gen.list_of_train_examples)
        # number_of_batches = math.ceil(number_of_seqs / batch_size)

        # ** new -
        number_of_seqs = len(train_data_loader.sampler)
        number_of_batches = len(train_data_loader.batch_sampler)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + str(test_split))

        # if args.upload is True:
        wandb.init(project=args.project, group=args.group,
                   name="split: " + str(test_split), entity=args.entity,  # ** we added entity, mode
                   mode=args.wandb_mode)
        # delattr(args, 'test_split')
        wandb.config.update(args, allow_val_change=True)
        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # ** new -
        schedular = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-2,
                                      threshold_mode='abs', verbose=True)
        best_acc = 0
        best_results = {'Acc gesture': 0, 'epoch': 0}
        steps_no_improve = 0
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0

            # ** old -
            # while batch_gen.has_next():
            #     batch_input, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
            #     batch_input, batch_target_gestures, mask = batch_input.to(self.device), batch_target_gestures.to(
            #       self.device), mask.to(self.device)

            # ** new -
            for batch in train_data_loader:
                batch_input, batch_target, lengths, mask = batch
                batch_input = {input: data.to(self.device) for input, data in batch_input.items()}
                batch_target = {input: data.to(self.device) for input, data in batch_target.items()}
                batch_target_gestures = batch_target['gestures']
                mask = mask.to(self.device)

                optimizer.zero_grad()
                # ** old -
                # lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')

                # ** new - received as part of batch
                lengths = lengths.to(dtype=torch.int64).to(device='cpu')

                # ** old -
                # predictions1 = self.model(batch_input, lengths)
                # ** new -
                predictions1 = self.model(batch_input, lengths, mask)
                # predictions1 = predictions1[-1].permute(0, 2, 1)

                # ** old -
                # loss = 0
                # for p in predictions1:
                #     loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]),
                #                     batch_target_gestures.view(-1))

                # ** new -
                # losses = []
                for task_num, task_predictions in enumerate(predictions1):
                    task_loss = 0
                    for p_stage in task_predictions:
                        task_loss += self.ce(
                            p_stage.transpose(2, 1).contiguous().view(-1, self.num_classes_list[task_num]),
                            batch_target[self.task[task_num]].view(-1))
                        task_loss += loss_factor * torch.mean(
                            torch.clamp(
                                self.mse(nn.functional.log_softmax(p_stage[:, :, 1:], dim=1),
                                         nn.functional.log_softmax(p_stage.detach()[:, :, :-1], dim=1)),
                                min=0, max=T) * mask[:, 1:, 0:p_stage.shape[1]].permute(0, 2, 1))
                    if task_num == 0:
                        loss = task_loss
                    else:
                        loss = loss + task_loss
                    # losses.append(task_loss)
                # epoch_loss = 0
                # for task_loss in losses:
                #     task_loss.backward()
                #     epoch_loss += task_loss.item()
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                # _, predicted1 = torch.max(predictions1[-1], 1)
                _, predicted1 = torch.max(predictions1[0][-1], 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][
                                                               :lengths[i]].squeeze()).float().sum().item()
                    total1 += lengths[i]

                pbar.update(1)

            # ** new -
            acc = correct1 / total1
            schedular.step(acc)

            # ** old -
            # batch_gen.reset()

            pbar.close()

            # ** old:
            # if not self.debugging:
            #     torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            #     torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # ** new is at end of loop

            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            # ** old -
            # print(colored(dt_string, 'green',
            #               attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
            #                                                                                           epoch_loss / len(
            #                                                                                               batch_gen.list_of_train_examples),
            #                                                                                           float(
            #                                                                                               correct1) / total1))
            # train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
            #                  "train acc": float(correct1) / total1}

            # ** new -
            print(colored(
                dt_string, 'green',
                attrs=['bold']) + f"  [epoch {epoch + 1}: train loss = {epoch_loss / number_of_seqs},   "
                                  f"train acc = {float(correct1) / total1}")
            train_results = {"epoch": epoch, "train loss": epoch_loss / number_of_seqs,
                             "train acc": float(correct1) / total1}

            # if args.upload: # **controlled by wandb mode
            wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch + 1) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}

                # ** old -
                # results.update(self.evaluate(eval_dict, batch_gen))

                # ** new -
                results.update(self.evaluate(eval_dict, test_data_loader, list_of_vids))
                eval_results_list.append(results)

                if results['Acc gesture'] > best_results['Acc gesture'] + 5e-3:
                    best_results.update(results)
                    best_results['epoch'] = epoch

                # if args.upload is True:  # **controlled by wandb mode
                wandb.log(results)

            # **new:
            if acc > best_acc + 1e-2:
                torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, "model.h5"))
                torch.save(optimizer.state_dict(), os.path.join(wandb.run.dir, "optimizer.h5"))

                best_acc = acc
                steps_no_improve = 0
            else:
                steps_no_improve += 1
                if steps_no_improve >= early_stop:
                    break

        wandb.log({f'best_{k}': v for k, v in best_results.items()})
        wandb.finish()
        return eval_results_list, train_results_list, best_results

    # ** old:
    # def evaluate(self, eval_dict, batch_gen):
    # ** new:
    def evaluate(self, eval_dict, test_data_loader, list_of_vids):
        # ** old:
        # device = eval_dict["device"]
        # features_path = eval_dict["features_path"]
        results = {}
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gestures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)

            recognition1_list = []

            # ** old
            # list_of_vids = batch_gen.list_of_valid_examples
            # for seq in list_of_vids:
            # print vid

            # ** new
            for batch in test_data_loader:
                batch_input, batch_target, lengths, mask = batch
                batch_input = {input: data.to(self.device) for input, data in batch_input.items()}
                batch_target = {input: data.to(self.device) for input, data in batch_target.items()}
                batch_target_gestures = batch_target['gestures']
                mask = mask.to(self.device)

                # ** old

                # features = np.load(features_path + seq.split('.')[0] + '.npy')
                # features = features[:, ::sample_rate]
                # input_x = torch.tensor(features, dtype=torch.float)
                # input_x.unsqueeze_(0)
                # input_x = input_x.to(device)
                # predictions1 = self.model(input_x, torch.tensor([features.shape[1]]))
                # predictions1 = predictions1[0].unsqueeze_(0)

                # **new
                predictions1 = self.model(batch_input, lengths, mask)
                predictions1 = predictions1[0][-1].permute(0, 2, 1)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                # **old
                # _, predicted1 = torch.max(predictions1[-1].data, 1)

                # ** new
                _, predicted1 = torch.max(predictions1, 2)

                # ** old
                # predicted1 = predicted1.squeeze()

                # for i in range(len(lengths)):
                #     correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][:lengths[i]].squeeze()).float().sum().item()

                for j in range(len(lengths)):
                    sur_prediction = predicted1[j][:lengths[j]]
                    recognition1 = []
                    for i in range(len(sur_prediction)):
                        recognition1 = np.concatenate((recognition1, [list(actions_dict_gestures.keys())[
                                                                          list(actions_dict_gestures.values()).index(
                                                                              sur_prediction[
                                                                                  i].item())]] * sample_rate))
                    recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            results.update(results1)

            self.model.train()
            return results

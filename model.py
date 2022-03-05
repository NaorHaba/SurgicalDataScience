# Created by Adam Goldbraikh - Scalpel Lab Technion

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn


# # input size = (B, F(2048), N(SURGERY))

# #MULTI TASK
class MS_TCN_PP(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
        super(MS_TCN_PP, self).__init__()
        stages = []
        dilations_inc = [2 ** i for i in range(num_layers)]
        dilations_dec = [2 ** i for i in range(num_layers - 1, -1, -1)]
        dilations = list(zip(dilations_inc, dilations_dec))
        dilation_layer = DualDilatedResidualLayer
        in_dim = dim
        self.softmax = nn.Softmax(dim=1)
        for i in range(num_stages):
            if i != 0:
                in_dim = sum(num_classes)
                dilations = dilations_inc
                dilation_layer = DilatedResidualLayer
            stages.append(SS_TCN(dilations, dilation_layer, num_f_maps, in_dim, num_classes, **kw))
        self.stages = nn.ModuleList(stages)

    def forward(self, x, mask):
        out = [out_task * mask for out_task in self.stages[0](x.clone().detach(), mask)]
        outputs = [[out_task.unsqueeze(0)] for out_task in out]
        for s in self.stages[1:]:
            out = [self.softmax(out_task) * mask for out_task in out]
            out = torch.cat(out, dim=1)
            out = [out_task * mask for out_task in s(out, mask)]
            outputs = [outputs[i] + [out_task.unsqueeze(0)] for i, out_task in enumerate(out)]
        return [torch.cat(outputs_task, dim=0) for outputs_task in outputs]


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
        super(MS_TCN, self).__init__()
        stages = []
        in_dim = dim
        dilations = [2 ** i for i in range(num_layers)]
        self.softmax = nn.Softmax(dim=1)
        for i in range(num_stages):
            if i != 0:
                in_dim = sum(num_classes)
            stages.append(SS_TCN(dilations, DilatedResidualLayer, num_f_maps, in_dim, num_classes, **kw))
        self.stages = nn.ModuleList(stages)
        # self.stages = stages

    def forward(self, x, mask):
        out = [out_task * mask for out_task in self.stages[0](x.clone().detach(), mask)]
        outputs = [[out_task.unsqueeze(0)] for out_task in out]
        for s in self.stages[1:]:
            out = [self.softmax(out_task) * mask for out_task in out]
            out = torch.cat(out, dim=1)
            out = [out_task * mask for out_task in s(out, mask)]
            outputs = [outputs[i] + [out_task.unsqueeze(0)] for i, out_task in enumerate(out)]
        return [torch.cat(outputs_task, dim=0) for outputs_task in outputs]


class SS_TCN(nn.Module):
    def __init__(self, dilations, dilated_layer, num_f_maps, dim, num_classes, **kw):
        super(SS_TCN, self).__init__()
        self.gate_in = nn.Conv1d(dim, num_f_maps, 1)
        self.stage = nn.Sequential(
            *[
                dilated_layer(dilation, num_f_maps, num_f_maps, **kw) for dilation in dilations
            ],
        )
        convs = [nn.Conv1d(num_f_maps, num_classes[i], 1) for i in range(len(num_classes))]
        self.gate_out = nn.ModuleList(convs)

    def forward(self, x, mask):
        out = self.gate_in(x) * mask
        for sub_stage in self.stage:
            out = sub_stage(out, mask)
        outputs = [g(out.clone().detach()) * mask for g in self.gate_out]
        return outputs


# #
# one task
# class MS_TCN_PP(nn.Module):
#     def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
#         super(MS_TCN_PP, self).__init__()
#         stages = []
#         dilations_inc = [2 ** i for i in range(num_layers)]
#         dilations_dec = [2 ** i for i in range(num_layers - 1, -1, -1)]
#         dilations = list(zip(dilations_inc, dilations_dec))
#         dilation_layer = DualDilatedResidualLayer
#         in_dim = dim
#         self.softmax = nn.Softmax(dim=1)
#         for i in range(num_stages):
#             if i != 0:
#                 in_dim = num_classes
#                 dilations = dilations_inc
#                 dilation_layer = DilatedResidualLayer
#             # stages.append(nn.Sequential(SS_TCN(dilations, dilation_layer, num_f_maps, in_dim, num_classes, **kw),
#             #                             nn.Softmax()))
#             stages.append(SS_TCN(dilations, dilation_layer, num_f_maps, in_dim, num_classes, **kw))
#         # self.stages = nn.ModuleList(stages)
#         self.stages = nn.ModuleList(stages)
#
#
#     def forward(self, x, mask):
#         out = self.stages[0](x, mask) * mask
#         outputs = [out.unsqueeze(0)]
#         for s in self.stages[1:]:
#             out = s(self.softmax(out) * mask, mask) * mask
#             outputs.append(out.unsqueeze(0))
#         return torch.cat(outputs, dim=0)
#         # outputs.append(out)
#         # return torch.cat(outputs)  # i changed this and changed MASK


# class MS_TCN(nn.Module):
#     def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, **kw):
#         super(MS_TCN, self).__init__()
#         stages = []
#         in_dim = dim
#         dilations = [2 ** i for i in range(num_layers)]
#         self.softmax = nn.Softmax(dim=1)
#         for i in range(num_stages):
#             if i != 0:
#                 in_dim = num_classes
#             stages.append(SS_TCN(dilations, DilatedResidualLayer, num_f_maps, in_dim, num_classes, **kw))
#         self.stages = nn.ModuleList(stages)
#         # self.stages = stages
#
#     def forward(self, x, mask):
#         out = self.stages[0](x, mask) * mask
#         outputs = [out.unsqueeze(0)]
#         for s in self.stages[1:]:
#             out = s(self.softmax(out) * mask, mask) * mask
#             outputs.append(out.unsqueeze(0))
#         return torch.cat(outputs, dim=0)
#
# class SS_TCN(nn.Module):
#     def __init__(self, dilations, dilated_layer, num_f_maps, dim, num_classes, **kw):
#         super(SS_TCN, self).__init__()
#         self.gate_in = nn.Conv1d(dim, num_f_maps, 1)
#         self.stage = nn.Sequential(
#             *[
#                 dilated_layer(dilation, num_f_maps, num_f_maps, **kw) for dilation in dilations
#             ],
#         )
#         self.gate_out = nn.Conv1d(num_f_maps, num_classes, 1)
#
#     def forward(self, x, mask):
#         out = self.gate_in(x) * mask
#         for sub_stage in self.stage:
#             out = sub_stage(out, mask)
#         out = self.gate_out(out)
#         return out * mask
# #

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, activation=nn.ReLU, dropout=0.1):
        super(DilatedResidualLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            activation(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        out = self.layer(x)
        return (x + out) * mask


class DualDilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, activation=nn.ReLU, dropout=0.1):
        super(DualDilatedResidualLayer, self).__init__()
        dilation_inc, dilation_dec = dilation
        self.conv_inc = nn.Conv1d(in_channels, out_channels, 3, padding=dilation_inc, dilation=dilation_inc)
        self.conv_dec = nn.Conv1d(in_channels, out_channels, 3, padding=dilation_dec, dilation=dilation_dec)
        self.conv_to_out = nn.Sequential(
            nn.Conv1d(2 * out_channels, out_channels, 1),
            activation(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Dropout(dropout)
        )

    ##FIXME
    def forward(self, x, mask):
        inc = self.conv_inc(x.clone().detach())
        dec = self.conv_dec(x.clone().detach())
        out = self.conv_to_out(torch.cat([inc, dec], dim=1))
        return (x + out) * mask


# if not implementing extra logic, same as ModuleList
class SurgeryModel(nn.Module):
    def __init__(self, feature_extractor, time_series_model):
        super().__init__()
        self.fe = feature_extractor
        self.ts = time_series_model

    def forward(self, x, lengths, mask):
        mask = mask.permute(0, 2, 1)
        mask = mask[:, 0, :].unsqueeze(1)
        features = self.fe(x)
        features = torch.cat(features, dim=-1)
        features = features.permute(0, 2, 1)
        features *= mask
        # packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        # packed_result = self.ts(packed_features)
        # result = pad_packed_sequence(packed_result)
        result = self.ts(features, mask)
        return result


class SeperateFeatureExtractor(nn.Module):
    def __init__(self, top_fe=None, side_fe=None, kinematics_fe=None):  # accepts 'nn.Identity'
        super().__init__()
        self.fe = {}
        if top_fe:
            self.top_fe = top_fe
            self.fe['top'] = self.top_fe
        if side_fe:
            self.side_fe = side_fe
            self.fe['side'] = self.side_fe
        if kinematics_fe:
            self.kinematics_fe = kinematics_fe
            self.fe['kinematics'] = self.kinematics_fe
        assert self.fe, 'must receive at least one feature extractor'

    def forward(self, x):
        features = []
        for key in self.fe:
            try:
                features.append(self.fe[key](x[key]))
            except KeyError:
                raise RuntimeError(f"received feature extractor for {key} but it is not present in x")
        return features  # FIXME

# MODEL
# init - FE_model, TS_model
# forward -> return TS_model(FE_model)

# FE_model_separate
# init - T_fe = None, S_fe = None, K_fe = None (accepts 'identity')
# all_fe = {}
# if T_fe: all_fe[top] = T_fe ...
# forward:
# for key, fe in all_fe.items(): fe(input[key])
# return concat(features)

# ^ input = {"TOP": OPTIONAL, "SIDE": OPTIONAL, "KINEMATICS": OPTIONAL}


# class Trainer:
#     def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
#         self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
#         self.ce = nn.CrossEntropyLoss(ignore_index=-100)
#         self.mse = nn.MSELoss(reduction='none')
#         self.num_classes = num_classes
#
#     def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
#         self.model.train()
#         self.model.to(device)
#         optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         for epoch in range(num_epochs):
#             epoch_loss = 0
#             correct = 0
#             total = 0
#             while batch_gen.has_next():
#                 batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
#                 batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
#                 optimizer.zero_grad()
#                 predictions = self.model(batch_input, mask)
#
#                 loss = 0
#                 for p in predictions:
#                     loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
#                     loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
#
#                 epoch_loss += loss.item()
#                 loss.backward()
#                 optimizer.step()
#
#                 _, predicted = torch.max(predictions[-1].data, 1)
#                 correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
#                 total += torch.sum(mask[:, 0, :]).item()
#
#             batch_gen.reset()
#             torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
#             torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
#             print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
#                                                                float(correct)/total))
#
#     def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
#         self.model.eval()
#         with torch.no_grad():
#             self.model.to(device)
#             self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
#             file_ptr = open(vid_list_file, 'r')
#             list_of_vids = file_ptr.read().split('\n')[:-1]
#             file_ptr.close()
#             for vid in list_of_vids:
#                 print vid
#                 features = np.load(features_path + vid.split('.')[0] + '.npy')
#                 features = features[:, ::sample_rate]
#                 input_x = torch.tensor(features, dtype=torch.float)
#                 input_x.unsqueeze_(0)
#                 input_x = input_x.to(device)
#                 predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
#                 _, predicted = torch.max(predictions[-1].data, 1)
#                 predicted = predicted.squeeze()
#                 recognition = []
#                 for i in range(len(predicted)):
#                     recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
#                 f_name = vid.split('/')[-1].split('.')[0]
#                 f_ptr = open(results_dir + "/" + f_name, "w")
#                 f_ptr.write("### Frame level recognition: ###\n")
#                 f_ptr.write(' '.join(recognition))
#                 f_ptr.close()
#
#

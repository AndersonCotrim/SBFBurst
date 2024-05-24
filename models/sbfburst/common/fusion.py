# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSum(nn.Module):
    """ Performs adaptive weighted-sum fusion to merge the input embeddings of the burst images """

    def __init__(self, input_dim, offset_feat_dim,
                 num_offset_feat_extractor_res=1, num_weight_predictor_res=1, use_offset=True, offset_modulo=None,
                 ref_offset_noise=0.0, softmax=True, use_base_frame=False,
                 use_bn=False, activation='lrelu', ):
        super().__init__()
        self.use_offset = use_offset
        self.offset_modulo = offset_modulo
        self.ref_offset_noise = ref_offset_noise
        self.softmax = softmax
        self.use_base_frame = use_base_frame

        project_dim = input_dim

        offset_feat_extractor = []
        offset_feat_extractor.append(conv_block(2, offset_feat_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                activation=activation))

        for _ in range(num_offset_feat_extractor_res):
            offset_feat_extractor.append(ResBlock(offset_feat_dim, offset_feat_dim, stride=1,
                                                  batch_norm=use_bn, activation=activation))
        self.offset_feat_extractor = nn.Sequential(*offset_feat_extractor)

        weight_predictor = []
        weight_predictor.append(conv_block(project_dim * 2 + offset_feat_dim * use_offset, 2 * project_dim, 3,
                                           stride=1, padding=1, batch_norm=use_bn, activation=activation))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                             batch_norm=use_bn, activation=activation))

        weight_predictor.append(conv_block(2 * project_dim, input_dim, 3, stride=1, padding=1,
                                           batch_norm=use_bn,
                                           activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, x):
        ref_feat = x['ref_feat']
        oth_feat = x['oth_feat']
        offsets = x['offsets']

        assert ref_feat.dim() == 5
        if ref_feat.shape[1] != 1:
            ref_feat = ref_feat[:, :1, ...].contiguous()

        shape = ref_feat.shape

        all_feat = torch.cat((ref_feat, oth_feat), dim=1)

        all_feat_proj = all_feat

        # Select the base embeddings which is either the embeddings of the reference (first) image, or the mean
        # embedding over all images
        if getattr(self, 'use_base_frame', False):
            base_feat_proj = all_feat_proj[:, :1].contiguous()
        else:
            base_feat_proj = all_feat_proj.mean(dim=1, keepdim=True)

        # Compute the residual between the base embeddings and other embeddings
        feat_diff_proj = all_feat_proj - base_feat_proj
        feat_diff_proj = feat_diff_proj.view(-1, *feat_diff_proj.shape[-3:])
        base_feat_proj = base_feat_proj.expand(-1, all_feat.shape[1], -1, -1, -1).contiguous().view(-1,
                                                                                                    *base_feat_proj.shape[
                                                                                                     -3:])
        weight_pred_in = [base_feat_proj, feat_diff_proj]

        if getattr(self, 'use_offset', True):
            if getattr(self, 'ref_offset_noise', 0.0) > 0.0:
                # If ref_offset_noise > 0, add some noise to the offests of reference image (originally all zeros) so
                # that the network cannot learn to use only the reference frame embeddings
                offsets_base = torch.rand((shape[0], 1, 2, *shape[-2:])).float().to(ref_feat.device) * 2 * \
                               getattr(self, 'ref_offset_noise', 0.0) - getattr(self, 'ref_offset_noise', 0.0)
            else:
                offsets_base = torch.zeros((shape[0], 1, 2, *shape[-2:])).float().to(ref_feat.device)

            offsets_all = torch.cat((offsets_base, offsets), dim=1)
            offsets_all = offsets_all.view(-1, *offsets_all.shape[-3:])

            # Since we are only interested in sub-pixel sampling location, compute a modulo of the offsets
            if getattr(self, 'offset_modulo', None) is not None:
                offsets_all = offsets_all % self.offset_modulo

            offsets_feat = self.offset_feat_extractor(offsets_all)
            weight_pred_in.append(offsets_feat)

        weight_pred_in = torch.cat(weight_pred_in, dim=1)

        # Compute attention weights
        weights = self.weight_predictor(weight_pred_in)
        weights = weights.view(shape[0], -1, *weights.shape[-3:])

        # Normalize the weights
        if self.softmax:
            weights_norm = F.softmax(weights, dim=1)
        else:
            weights_norm = F.relu(weights)
            weights_norm = weights_norm / (weights_norm.sum(dim=1, keepdim=True) + 1e-12)

        # Perform fusion
        fused_feat = (all_feat * weights_norm).sum(dim=1)

        out = {'fused_enc': fused_feat, 'fusion_weights': weights_norm}
        return out


def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def get_attention(attention_type, num_channels=None):
    if attention_type == 'none':
        return None
    else:
        raise Exception('Unknown attention {}'.format(attention_type))


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))

    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv2 = conv_block(planes, planes, kernel_size=3, padding=1, dilation=dilation, batch_norm=batch_norm,
                                activation='none', padding_mode=padding_mode)

        self.downsample = downsample
        self.stride = stride

        self.activation = get_activation(activation, num_channels=planes)
        self.attention = get_attention(attention_type=attention, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention is not None:
            out = self.attention(out)

        out += residual

        out = self.activation(out)

        return out

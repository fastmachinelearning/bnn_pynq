# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
from .tensor_norm import TensorNorm
from .common import CommonWeightQuant, CommonActQuant


class CNV(Module):

    def __init__(self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch,
                 cnv_out_ch_pool, int_fc_feat, last_fc_in_feat, pool_size, kern_size):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(QuantIdentity( # for Q1.7 input format
            act_quant=CommonActQuant,
            bit_width=in_bit_width,
            min_val=- 1.0,
            max_val=1.0 - 2.0 ** (-7),
            narrow_range=False,
            restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in cnv_out_ch_pool:
            self.conv_features.append(QuantConv2d(
                kernel_size=kern_size,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=pool_size))

        for in_features, out_features in int_fc_feat:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=last_fc_in_feat,
            out_features=num_classes,
            bias=False,
            weight_quant=CommonWeightQuant,
            weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())
        
        for m in self.modules():
          if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
            torch.nn.init.uniform_(m.weight.data, -1, 1)


    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv(cfg):
    weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
    act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    in_bit_width = cfg.getint('QUANT', 'IN_BIT_WIDTH')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    in_channels = cfg.getint('MODEL', 'IN_CHANNELS')
    cnv_out_ch_pool = cfg.get('MODEL', 'CNV_OUT_CH_POOL')
    int_fc_feat = cfg.get('MODEL', 'INTERMEDIATE_FC_FEATURES')
    cnv_out_ch_pool = [eval(pair) for pair in cnv_out_ch_pool.split('\n')]
    int_fc_feat = [eval(pair) for pair in int_fc_feat.split('\n')]
    last_fc_in_feat = cfg.getint('MODEL', 'LAST_FC_IN_FEATURES')
    pool_size = cfg.getint('MODEL', 'POOL_SIZE')
    kern_size = cfg.getint('MODEL', 'KERNEL_SIZE')

    net = CNV(weight_bit_width=weight_bit_width,
              act_bit_width=act_bit_width,
              in_bit_width=in_bit_width,
              num_classes=num_classes,
              in_ch=in_channels,
              cnv_out_ch_pool=cnv_out_ch_pool,
              int_fc_feat=int_fc_feat,
              last_fc_in_feat=last_fc_in_feat,
	      pool_size=pool_size,
              kern_size=kern_size)
    return net


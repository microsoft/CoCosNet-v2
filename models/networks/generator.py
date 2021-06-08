# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from models.networks.base_network import BaseNetwork
from models.networks.architecture import SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf
        self.sw, self.sh = self.compute_latent_vector_size(opt)
        ic = 4*3+opt.label_nc
        self.fc = nn.Conv2d(ic, 8 * nf, 3, padding=1)
        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.up_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        final_nc = nf
        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        num_up_layers = 5
        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)
        return sw, sh

    def forward(self, input, warp_out=None):
        seg = torch.cat((F.interpolate(warp_out[0], size=(512, 512)), F.interpolate(warp_out[1], size=(512, 512)), F.interpolate(warp_out[2], size=(512, 512)), warp_out[3], input), dim=1)
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

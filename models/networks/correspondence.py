# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.util as util
from models.networks.base_network import BaseNetwork
from models.networks.architecture import ResidualBlock
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResnetBlock
"""patch match"""
from models.networks.patch_match import PatchMatchGRU
from models.networks.ops import *


def match_kernel_and_pono_c(feature, match_kernel, PONO_C, eps=1e-10):
    b, c, h, w = feature.size()
    if match_kernel == 1:
        feature = feature.view(b, c, -1)
    else:
        feature = F.unfold(feature, kernel_size=match_kernel, padding=int(match_kernel//2))
    dim_mean = 1 if PONO_C else -1
    feature = feature - feature.mean(dim=dim_mean, keepdim=True)
    feature_norm = torch.norm(feature, 2, 1, keepdim=True) + eps
    feature = torch.div(feature, feature_norm)
    return feature.view(b, -1, h, w)


"""512x512"""
class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = opt.featEnc_kernel
        pw = int((kw-1)//2)
        nf = opt.nef
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, nf, 3, stride=1, padding=pw))
        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 1, nf * 2, 3, 1, 1)),
            ResidualBlock(nf * 2, nf * 2),
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 2, nf * 4, kw, stride=2, padding=pw)),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kw, stride=2, padding=pw)),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf * 4, nf * 4, kw, stride=2, padding=pw)),
            ResidualBlock(nf * 4, nf * 4),
        )
        self.head_0 = SPADEResnetBlock(nf * 4, nf * 4, opt)
        self.G_middle_0 = SPADEResnetBlock(nf * 4, nf * 4, opt)
        self.G_middle_1 = SPADEResnetBlock(nf * 4, nf * 2, opt)
        self.G_middle_2 = SPADEResnetBlock(nf * 2, nf * 1, opt)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input, seg):
        # 512
        x1 = self.layer1(input)
        # 512
        x2 = self.layer2(self.actvn(x1))
        # 256
        x3 = self.layer3(self.actvn(x2))
        # 128
        x4 = self.layer4(self.actvn(x3))
        # 64
        x5 = self.layer5(self.actvn(x4))
        # bottleneck
        x6 = self.head_0(x5, seg)
        # 128
        x7 = self.G_middle_0(self.up(x6) + x4, seg)
        # 256
        x8 = self.G_middle_1(self.up(x7) + x3, seg)
        # 512
        x9 = self.G_middle_2(self.up(x8) + x2, seg)
        return [x6, x7, x8, x9]

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class NoVGGHPMCorrespondence(BaseNetwork):
    def __init__(self, opt):
        self.opt = opt
        super().__init__()
        opt.spade_ic = opt.semantic_nc
        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt)
        opt.spade_ic = 3 + opt.semantic_nc
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt)
        del opt.spade_ic
        self.batch_size = opt.batchSize
        """512x512"""
        feature_channel = opt.nef
        self.phi_0 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_2 = nn.Conv2d(in_channels=feature_channel*2, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.phi_3 = nn.Conv2d(in_channels=feature_channel, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_0 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_1 = nn.Conv2d(in_channels=feature_channel*4, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_2 = nn.Conv2d(in_channels=feature_channel*2, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.theta_3 = nn.Conv2d(in_channels=feature_channel, out_channels=feature_channel, kernel_size=1, stride=1, padding=0)
        self.patch_match = PatchMatchGRU(opt)

    """512x512"""
    def multi_scale_patch_match(self, f1, f2, ref, hierarchical_scale, pre=None, real_img=None):
        if hierarchical_scale == 0:
            y_cycle = None
            scale = 64
            batch_size, channel, feature_height, feature_width = f1.size()
            ref = F.avg_pool2d(ref, 8, stride=8)
            ref = ref.view(batch_size, 3, scale * scale)
            f1 = f1.view(batch_size, channel, scale * scale)
            f2 = f2.view(batch_size, channel, scale * scale)
            matmul_result = torch.matmul(f1.permute(0, 2, 1), f2)/self.opt.temperature
            mat = F.softmax(matmul_result, dim=-1)
            y = torch.matmul(mat, ref.permute(0, 2, 1))
            if self.opt.phase is 'train' and self.opt.weight_warp_cycle > 0:
                mat_cycle = F.softmax(matmul_result.transpose(1, 2), dim=-1)
                y_cycle = torch.matmul(mat_cycle, y)
                y_cycle = y_cycle.permute(0, 2, 1).view(batch_size, 3, scale, scale)
            y = y.permute(0, 2, 1).view(batch_size, 3, scale, scale)
            return mat, y, y_cycle
        if hierarchical_scale == 1:
            scale = 128
            with torch.no_grad():
                batch_size, channel, feature_height, feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window ** 2)
                topk_inds = torch.topk(pre, topk_num, dim=-1)[-1]
                inds = topk_inds.permute(0, 2, 1).view(batch_size, topk_num, (scale//2), (scale//2)).float()
                offset_x, offset_y = inds_to_offset(inds)
                dx = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=1).expand(-1, search_window).contiguous().view(-1) - centering
                dy = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=0).expand(search_window, -1).contiguous().view(-1) - centering
                dx = dx.view(1, search_window ** 2, 1, 1) * dilation
                dy = dy.view(1, search_window ** 2, 1, 1) * dilation
                offset_x_up = F.interpolate((2 * offset_x + dx), scale_factor=2)
                offset_y_up = F.interpolate((2 * offset_y + dy), scale_factor=2)
            ref = F.avg_pool2d(ref, 4, stride=4)
            ref = ref.view(batch_size, 3, scale * scale)
            mat, y = self.patch_match(f1, f2, ref, offset_x_up, offset_y_up)
            y = y.view(batch_size, 3, scale, scale)
            return mat, y
        if hierarchical_scale == 2:
            scale = 256
            with torch.no_grad():
                batch_size, channel, feature_height, feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window ** 2)
                topk_inds = pre[:, :, :topk_num]
                inds = topk_inds.permute(0, 2, 1).view(batch_size, topk_num, (scale//2), (scale//2)).float()
                offset_x, offset_y = inds_to_offset(inds)
                dx = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=1).expand(-1, search_window).contiguous().view(-1) - centering
                dy = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=0).expand(search_window, -1).contiguous().view(-1) - centering
                dx = dx.view(1, search_window ** 2, 1, 1) * dilation
                dy = dy.view(1, search_window ** 2, 1, 1) * dilation
                offset_x_up = F.interpolate((2 * offset_x + dx), scale_factor=2)
                offset_y_up = F.interpolate((2 * offset_y + dy), scale_factor=2)
            ref = F.avg_pool2d(ref, 2, stride=2)
            ref = ref.view(batch_size, 3, scale * scale)
            mat, y = self.patch_match(f1, f2, ref, offset_x_up, offset_y_up)
            y = y.view(batch_size, 3, scale, scale)
            return mat, y
        if hierarchical_scale == 3:
            scale = 512
            with torch.no_grad():
                batch_size, channel, feature_height, feature_width = f1.size()
                topk_num = 1
                search_window = 4
                centering = 1
                dilation = 2
                total_candidate_num = topk_num * (search_window ** 2)
                topk_inds = pre[:, :, :topk_num]
                inds = topk_inds.permute(0, 2, 1).view(batch_size, topk_num, (scale//2), (scale//2)).float()
                offset_x, offset_y = inds_to_offset(inds)
                dx = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=1).expand(-1, search_window).contiguous().view(-1) - centering
                dy = torch.arange(search_window, dtype=topk_inds.dtype, device=topk_inds.device).unsqueeze_(dim=0).expand(search_window, -1).contiguous().view(-1) - centering
                dx = dx.view(1, search_window ** 2, 1, 1) * dilation
                dy = dy.view(1, search_window ** 2, 1, 1) * dilation
                offset_x_up = F.interpolate((2 * offset_x + dx), scale_factor=2)
                offset_y_up = F.interpolate((2 * offset_y + dx), scale_factor=2)
            ref = ref.view(batch_size, 3, scale * scale)
            mat, y = self.patch_match(f1, f2, ref, offset_x_up, offset_y_up)
            y = y.view(batch_size, 3, scale, scale)
            return mat, y

    def forward(self, ref_img, real_img, seg_map, ref_seg_map):
        corr_out = {}
        seg_input = seg_map
        adaptive_feature_seg = self.adaptive_model_seg(seg_input, seg_input)
        ref_input = torch.cat((ref_img, ref_seg_map), dim=1)
        adaptive_feature_img = self.adaptive_model_img(ref_input, ref_input)
        for i in range(len(adaptive_feature_seg)):
            adaptive_feature_seg[i] = util.feature_normalize(adaptive_feature_seg[i])
            adaptive_feature_img[i] = util.feature_normalize(adaptive_feature_img[i])
        if self.opt.isTrain and self.opt.weight_novgg_featpair > 0:
            real_input = torch.cat((real_img, seg_map), dim=1)
            adaptive_feature_img_pair = self.adaptive_model_img(real_input, real_input)
            loss_novgg_featpair = 0
            weights = [1.0, 1.0, 1.0, 1.0]
            for i in range(len(adaptive_feature_img_pair)):
                adaptive_feature_img_pair[i] = util.feature_normalize(adaptive_feature_img_pair[i])
                loss_novgg_featpair += F.l1_loss(adaptive_feature_seg[i], adaptive_feature_img_pair[i]) * weights[i]
            corr_out['loss_novgg_featpair'] = loss_novgg_featpair * self.opt.weight_novgg_featpair
        cont_features = adaptive_feature_seg
        ref_features = adaptive_feature_img
        theta = []
        phi = []
        """512x512"""
        theta.append(match_kernel_and_pono_c(self.theta_0(cont_features[0]), self.opt.match_kernel, self.opt.PONO_C))
        theta.append(match_kernel_and_pono_c(self.theta_1(cont_features[1]), self.opt.match_kernel, self.opt.PONO_C))
        theta.append(match_kernel_and_pono_c(self.theta_2(cont_features[2]), self.opt.match_kernel, self.opt.PONO_C))
        theta.append(match_kernel_and_pono_c(self.theta_3(cont_features[3]), self.opt.match_kernel, self.opt.PONO_C))
        phi.append(match_kernel_and_pono_c(self.phi_0(ref_features[0]), self.opt.match_kernel, self.opt.PONO_C))
        phi.append(match_kernel_and_pono_c(self.phi_1(ref_features[1]), self.opt.match_kernel, self.opt.PONO_C))
        phi.append(match_kernel_and_pono_c(self.phi_2(ref_features[2]), self.opt.match_kernel, self.opt.PONO_C))
        phi.append(match_kernel_and_pono_c(self.phi_3(ref_features[3]), self.opt.match_kernel, self.opt.PONO_C))
        ref = ref_img
        ys = []
        m = None
        for i in range(len(theta)):
            if i == 0:
                m, y, y_cycle = self.multi_scale_patch_match(theta[i], phi[i], ref, i, pre=m)
                if y_cycle is not None:
                    corr_out['warp_cycle'] = y_cycle
            else:
                m, y  = self.multi_scale_patch_match(theta[i], phi[i], ref, i, pre=m)
            ys.append(y)
        corr_out['warp_out'] = ys
        return corr_out

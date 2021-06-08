# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from models.networks.convgru import BasicUpdateBlock
from models.networks.ops import *


"""patch match"""
class Evaluate(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.filter_size = 3
        self.temperature = temperature

    def forward(self, left_features, right_features, offset_x, offset_y):
        device = left_features.get_device()
        batch_size, num, height, width = offset_x.size()
        channel = left_features.size()[1]
        matching_inds = offset_to_inds(offset_x, offset_y)
        matching_inds = matching_inds.view(batch_size, num, height * width).permute(0, 2, 1).long()
        base_batch = torch.arange(batch_size).to(device).long() * (height * width)
        base_batch = base_batch.view(-1, 1, 1)
        matching_inds_add_base = matching_inds + base_batch
        right_features_view = right_features
        match_cost = []
        # using A[:, idx]
        for i in range(matching_inds_add_base.size()[-1]):
            idx = matching_inds_add_base[:, :, i]
            idx = idx.contiguous().view(-1)
            right_features_select = right_features_view[:, idx]
            right_features_select = right_features_select.view(channel, batch_size, -1).transpose(0, 1)
            match_cost_i = torch.sum(left_features * right_features_select, dim=1, keepdim=True) / self.temperature
            match_cost.append(match_cost_i)
        match_cost = torch.cat(match_cost, dim=1).transpose(1, 2)
        match_cost = F.softmax(match_cost, dim=-1)
        match_cost_topk, match_cost_topk_indices = torch.topk(match_cost, num//self.filter_size, dim=-1)
        matching_inds = torch.gather(matching_inds, -1, match_cost_topk_indices)
        matching_inds = matching_inds.permute(0, 2, 1).view(batch_size, -1, height, width).float()
        offset_x, offset_y = inds_to_offset(matching_inds)
        corr = match_cost_topk.permute(0, 2, 1)
        return offset_x, offset_y, corr


class PropagationFaster(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, offset_x, offset_y, propagation_type="horizontal"):
        device = offset_x.get_device()
        self.horizontal_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], offset_x.size()[2], 1)).to(device)
        self.vertical_zeros = torch.zeros((offset_x.size()[0], offset_x.size()[1], 1, offset_x.size()[3])).to(device)
        if propagation_type is "horizontal":
            offset_x = torch.cat((torch.cat((self.horizontal_zeros, offset_x[:, :, :, :-1]), dim=3),
                                  offset_x,
                                  torch.cat((offset_x[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)

            offset_y = torch.cat((torch.cat((self.horizontal_zeros, offset_y[:, :, :, :-1]), dim=3),
                                  offset_y,
                                  torch.cat((offset_y[:, :, :, 1:], self.horizontal_zeros), dim=3)), dim=1)

        else:
            offset_x = torch.cat((torch.cat((self.vertical_zeros, offset_x[:, :, :-1, :]), dim=2),
                                  offset_x,
                                  torch.cat((offset_x[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)

            offset_y = torch.cat((torch.cat((self.vertical_zeros, offset_y[:, :, :-1, :]), dim=2),
                                  offset_y,
                                  torch.cat((offset_y[:, :, 1:, :], self.vertical_zeros), dim=2)), dim=1)
        return offset_x, offset_y


class PatchMatchOnce(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.propagation = PropagationFaster()
        self.evaluate = Evaluate(opt.temperature)

    def forward(self, left_features, right_features, offset_x, offset_y):
        prob = random.random()
        if prob < 0.5:
            offset_x, offset_y = self.propagation(offset_x, offset_y, "horizontal")
            offset_x, offset_y, _ = self.evaluate(left_features, right_features, offset_x, offset_y)
            offset_x, offset_y = self.propagation(offset_x, offset_y, "vertical")
            offset_x, offset_y, corr = self.evaluate(left_features, right_features, offset_x, offset_y)
        else:
            offset_x, offset_y = self.propagation(offset_x, offset_y, "vertical")
            offset_x, offset_y, _ = self.evaluate(left_features, right_features, offset_x, offset_y)
            offset_x, offset_y = self.propagation(offset_x, offset_y, "horizontal")
            offset_x, offset_y, corr = self.evaluate(left_features, right_features, offset_x, offset_y)
        return offset_x, offset_y, corr


class PatchMatchGRU(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.patch_match_one_step = PatchMatchOnce(opt)
        self.temperature = opt.temperature
        self.iters = opt.iteration_count
        input_dim = opt.nef
        hidden_dim = 32
        norm = nn.InstanceNorm2d(hidden_dim, affine=False)
        relu = nn.ReLU(inplace=True)
        """
        concat left and right features
        """
        self.initial_layer = nn.Sequential(
            nn.Conv2d(input_dim*2, hidden_dim, kernel_size=3, padding=1, stride=1),
            norm,
            relu,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            norm,
            relu,
        )
        self.refine_net = BasicUpdateBlock()

    def forward(self, left_features, right_features, right_input, initial_offset_x, initial_offset_y):
        device = left_features.get_device()
        batch_size, channel, height, width = left_features.size()
        num = initial_offset_x.size()[1]
        initial_input = torch.cat((left_features, right_features), dim=1)
        hidden = self.initial_layer(initial_input)
        left_features = left_features.view(batch_size, -1, height * width)
        right_features = right_features.view(batch_size, -1, height * width)
        right_features_view = right_features.transpose(0, 1).contiguous().view(channel, -1)
        with torch.no_grad():
            offset_x, offset_y = initial_offset_x, initial_offset_y
        for it in range(self.iters):
            with torch.no_grad():
                offset_x, offset_y, corr = self.patch_match_one_step(left_features, right_features_view, offset_x, offset_y)
            """GRU refinement"""
            flow = torch.cat((offset_x, offset_y), dim=1)
            corr = corr.view(batch_size, -1, height, width)
            hidden, delta_offset_x, delta_offset_y = self.refine_net(hidden, corr, flow)
            offset_x = offset_x + delta_offset_x
            offset_y = offset_y + delta_offset_y
            with torch.no_grad():
                matching_inds = offset_to_inds(offset_x, offset_y)
                matching_inds = matching_inds.view(batch_size, num, height * width).permute(0, 2, 1).long()
                base_batch = torch.arange(batch_size).to(device).long() * (height * width)
                base_batch = base_batch.view(-1, 1, 1)
                matching_inds_plus_base = matching_inds + base_batch
        match_cost = []
        # using A[:, idx]
        for i in range(matching_inds_plus_base.size()[-1]):
            idx = matching_inds_plus_base[:, :, i]
            idx = idx.contiguous().view(-1)
            right_features_select = right_features_view[:, idx]
            right_features_select = right_features_select.view(channel, batch_size, -1).transpose(0, 1)
            match_cost_i = torch.sum(left_features * right_features_select, dim=1, keepdim=True) / self.temperature
            match_cost.append(match_cost_i)
        match_cost = torch.cat(match_cost, dim=1).transpose(1, 2)
        match_cost = F.softmax(match_cost, dim=-1)
        right_input_view = right_input.transpose(0, 1).contiguous().view(right_input.size()[1], -1)
        warp = torch.zeros_like(right_input)
        # using A[:, idx]
        for i in range(match_cost.size()[-1]):
            idx = matching_inds_plus_base[:, :, i]
            idx = idx.contiguous().view(-1)
            right_input_select = right_input_view[:, idx]
            right_input_select = right_input_select.view(right_input.size()[1], batch_size, -1).transpose(0, 1)
            warp = warp + right_input_select * match_cost[:, :, i].unsqueeze(dim=1)
        return matching_inds, warp

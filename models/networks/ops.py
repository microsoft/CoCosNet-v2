# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_1d_to_2d(index, base=64):
    x = index // base
    y = index % base
    return x,y


def convert_2d_to_1d(x, y, base=64):
    return x*base+y


def batch_meshgrid(shape, device):
    batch_size, _, height, width = shape
    x_range = torch.arange(0.0, width, device=device)
    y_range = torch.arange(0.0, height, device=device)
    x_coordinate, y_coordinate = torch.meshgrid(x_range, y_range)
    x_coordinate = x_coordinate.expand(batch_size, -1, -1).unsqueeze(1)
    y_coordinate = y_coordinate.expand(batch_size, -1, -1).unsqueeze(1)
    return x_coordinate, y_coordinate


def inds_to_offset(inds):
    """
    inds: b x number x h x w
    """
    shape = inds.size()
    device = inds.device
    x_coordinate, y_coordinate = batch_meshgrid(shape, device)
    batch_size, _, height, width = shape
    x = inds // width
    y = inds % width
    return x - x_coordinate, y - y_coordinate


def offset_to_inds(offset_x, offset_y):
    shape = offset_x.size()
    device = offset_x.device
    x_coordinate, y_coordinate = batch_meshgrid(shape, device)
    h, w = offset_x.size()[2:]
    x = torch.clamp(x_coordinate + offset_x, 0, h-1)
    y = torch.clamp(y_coordinate + offset_y, 0, w-1)
    return x * offset_x.size()[3] + y

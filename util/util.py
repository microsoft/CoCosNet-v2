# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import argparse
from argparse import Namespace
import torch
import numpy as np
import importlib
from PIL import Image



def feature_normalize(feature_in, eps=1e-10):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss


def mse_loss(input, target=0):
    return torch.mean((input - target)**2)


def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj
    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)
    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    if not os.path.exists(save_path):
        print('not find model :' + save_path + ', do not load model!')
        return net
    weights = torch.load(save_path)
    try:
        net.load_state_dict(weights)
    except KeyError:
        print('key error, not load!')
    except RuntimeError as err:
        print(err)
        net.load_state_dict(weights, strict=False)
        print('loaded with strict = False')
    print('Load from ' + save_path)
    return net


def print_current_errors(opt, epoch, i, num, errors, t):
    message = '(epoch: %d, iters: %d, finish: %.2f%%, time: %.3f) ' % (epoch, i, (i/num)*100.0, t)
    for k, v in errors.items():
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)
    print(message)
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

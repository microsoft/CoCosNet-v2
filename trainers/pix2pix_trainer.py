# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import copy
import sys
import torch
from models.pix2pix_model import Pix2PixModel
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self, enabled):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt, resume_epoch=0):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 1:
            self.pix2pix_model = torch.nn.DataParallel(self.pix2pix_model, device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model.to(opt.gpu_ids[0])
            self.pix2pix_model_on_one_gpu = self.pix2pix_model
        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr
            if opt.continue_train and opt.which_epoch == 'latest':
                try:
                    load_path = os.path.join(opt.checkpoints_dir, opt.name, 'optimizer.pth')
                    checkpoint = torch.load(load_path)
                    self.optimizer_G.load_state_dict(checkpoint['G'])
                    self.optimizer_D.load_state_dict(checkpoint['D'])
                except FileNotFoundError as err:
                    print(err)
                    print('Not find optimizer state dict: ' + load_path + '. Do not load optimizer!')

        self.last_data, self.last_netCorr, self.last_netG, self.last_optimizer_G = None, None, None, None
        self.g_losses = {}
        self.d_losses = {}
        self.scaler = GradScaler(enabled=self.opt.amp)

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, out = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        # g_loss.backward()
        self.scaler.scale(g_loss).backward()
        self.scaler.unscale_(self.optimizer_G)
        # self.optimizer_G.step()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.g_losses = g_losses
        self.out = out

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        GforD = {}
        GforD['fake_image'] = self.out['fake_image']
        GforD['adaptive_feature_seg'] = self.out['adaptive_feature_seg']
        GforD['adaptive_feature_img'] = self.out['adaptive_feature_img']
        d_losses = self.pix2pix_model(data, mode='discriminator', GforD=GforD)
        d_loss = sum(d_losses.values()).mean()
        # d_loss.backward()
        self.scaler.scale(d_loss).backward()
        self.scaler.unscale_(self.optimizer_D)
        # self.optimizer_D.step()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()
        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.out['fake_image']

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)
        if epoch == 'latest':
            torch.save({'G': self.optimizer_G.state_dict(), \
                        'D': self.optimizer_D.state_dict(), \
                        'lr':  self.old_lr,}, \
                        os.path.join(self.opt.checkpoints_dir, self.opt.name, 'optimizer.pth'))

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr
        if new_lr != self.old_lr:
            new_lr_G = new_lr
            new_lr_D = new_lr
        else:
            new_lr_G = self.old_lr
            new_lr_D = self.old_lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = new_lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn
import torch.nn.functional as F

from util.util import feature_normalize, mse_loss



class ContextualLoss_forward(nn.Module):
    '''
        input is Al, Bl, channel = 1, range ~ [0, 255]
    '''

    def __init__(self, opt):
        super(ContextualLoss_forward, self).__init__()
        self.opt = opt
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        '''
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        '''
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]
        feature_size = X_features.shape[2]

        # to normalized feature vectors
        if feature_centering:
            if self.opt.PONO:
                X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
                Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)
            else:
                X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
                Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

        # X_features = X_features - Y_features.mean(dim=1).unsqueeze(dim=1)
        # Y_features = Y_features - Y_features.mean(dim=1).unsqueeze(dim=1)

        X_features = feature_normalize(X_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size
        Y_features = feature_normalize(Y_features).view(batch_size, feature_depth, -1)  # batch_size * feature_depth * feature_size * feature_size

        # X_features = F.unfold(
        #     X_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2
        # Y_features = F.unfold(
        #     Y_features, kernel_size=self.opt.match_kernel, stride=1, padding=int(self.opt.match_kernel // 2))  # batch_size * feature_depth_new * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        # d_norm = d
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-3)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        loss = -torch.log(CX)

        # contextual loss per batch
        # loss = torch.mean(loss)
        return loss

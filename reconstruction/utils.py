import torch
import numpy as np
from torch import nn

def npComplexToTorch(kspace_np):

    # Converts a numpy complex to torch 
    kspace_real_torch=torch.from_numpy(kspace_np.real)
    kspace_imag_torch=torch.from_numpy(kspace_np.imag)
    kspace_torch = torch.stack([kspace_real_torch,kspace_imag_torch],dim=2)
    
    return kspace_torch

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S[self.feat_ind]
        feat_T = preds_T[self.feat_ind]
        feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
#         maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
#         loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        loss = self.criterion(feat_S,feat_T)

        return loss

class CriterionPairWiseforWholeFeatAfterPoolFeatureMaps(nn.Module):
    def __init__(self, scale, feat_ind):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPoolFeatureMaps, self).__init__()
        self.criterion = sim_dis_compute
        self.feat_ind = feat_ind
        self.scale = scale

    def forward(self, preds_S, preds_T):
        
        #feat_S = preds_S[self.feat_ind]
        #feat_T = preds_T[self.feat_ind]
        loss_layer = [] 
        loss_sum = 0.0
        for i in range(preds_S.shape[1]):
            #feat_S = preds_S[self.feat_ind]
            #feat_T = preds_T[self.feat_ind]
            feat_S = preds_S[:,i,:,:]
            feat_T = preds_T[:,i,:,:]
            feat_T.detach()

            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
#             maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
#             loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
            loss_sum += self.criterion(feat_S,feat_T)

        loss = loss_sum/preds_S.shape[1]

        return loss

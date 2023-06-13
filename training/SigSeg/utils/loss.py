"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from os import device_encoding
from pickle import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from collections import defaultdict  
from torch.autograd.function import Function
from utils.seg_loss import CE_DiceLoss


#Img Weighted Loss
class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), 3, (0,3), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()

        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda(3)

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda(0)
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0),dim=1),
                                          targets[i])
                                        
        return loss

class ImageFocalLoss(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageFocalLoss, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = FocalLoss(weight)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), 2, (0, 2), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()

        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda(0)

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda(1)


            loss+= self.nll_loss(inputs[i].view(-1, self.num_classes),
                                  targets[i].view(-1))

        return loss

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight


    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma *logpt

        loss = F.nll_loss(logpt, target, self.weight)
        return loss

#Cross Entroply NLL Loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        logging.info("Using Cross Entropy Loss")
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class MattingLoss(nn.Module):
    def __init__(self, loss_func_dict=None):
        super(MattingLoss, self).__init__()
        self.loss_func_dict = loss_func_dict
        if loss_func_dict is None:
            self.loss_func_dict = defaultdict(list)
            # self.loss_func_dict['semantic'].append(ImageBasedCrossEntropyLoss2d(3))  # FIXME 所有的损失函数 
            self.loss_func_dict['semantic'].append(CE_DiceLoss())  # FIXME 所有的损失函数 
            self.loss_func_dict['detail'].append(nn.L1Loss())
            self.loss_func_dict['fusion'].append(nn.L1Loss())
            self.loss_func_dict['fusion2'].append(nn.L1Loss())
            self.loss_func_dict['center'].append(CenterLoss(3, 3, size_average=True).to('cuda:0'))
        

    def compute_loss(self, logit_dict, label_dict):
        
        loss = {}
        # semantic  loss   
        # loss['semantic'] = self.loss_func_dict['semantic'][0](F.log_softmax(logit_dict['semantic'], dim=1),\
                                                                                                #   label_dict['trimap'].squeeze(1)) # FIXME 确定MSE的输入值范围
        loss['semantic'] = 5*self.loss_func_dict['semantic'][0](logit_dict['semantic'], label_dict['trimap']) # FIXME 确定MSE的输入值范围
        # loss['center'] = self.loss_func_dict['center'][0](logit_dict['semantic'], label_dict['trimap']) # FIXME 确定MSE的输入值范围

                                                                                                  
        
        # detail loss   FIXME 详情损失函数 加强对边界的约束 
        trimap = label_dict['trimap']
        mask = (trimap == 2).cuda(0).float() 
        logit_detail = logit_dict['detail'] * mask
        label_detail = label_dict['alpha'] * mask
        loss_detail = self.loss_func_dict['detail'][0](logit_detail, label_detail)
        loss_detail = loss_detail / (mask.mean() + 1e-6)
        loss['detail'] = 10 * loss_detail  # 详情损失环节加大约束程度  
        
        #fusion loss  加强对边界的约束 
        matte = logit_dict['alpha']
        alpha = label_dict['alpha']
        transition_mask = label_dict['trimap'] == 2  
        matte_boundary = torch.where(transition_mask, matte, alpha)
        # l1 loss
        loss_fusion_l1 = 3*self.loss_func_dict['fusion'][0](
            matte,
            alpha) + 4 * self.loss_func_dict['fusion'][0](matte_boundary, alpha)
        loss['funsion'] = loss_fusion_l1

        # composition loss 
        loss_fusion_comp = 3*self.loss_func_dict['fusion2'][0](
            matte * label_dict['image'],
            alpha* label_dict['image']) + 4* self.loss_func_dict['fusion2'][0](
                matte_boundary * label_dict['image'], alpha* label_dict['image'])  #FIXME  matte_boundary 的通道个数为32，是怎么回事？ 
        loss['composition'] = loss_fusion_comp
        loss['all'] =  loss['semantic']+ loss['detail']+ loss['funsion']+loss['composition']
        # loss['all'] =  loss['semantic']+ loss['funsion']+loss['composition']
        return loss  



class LSGANLoss(nn.Module):

    def __init__(self, device=None,  target_real_label=1.0, target_fake_label=0.0):
        super().__init__() 
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label)) 
        self.loss = nn.MSELoss()
        self.device = device


    def get_target_tensor(self, prediction, target_is_real):
        
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)


    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        target_tensor = target_tensor.to(self.device)
        loss = self.loss(prediction, target_tensor)
        return loss   


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feat, label):
        # print('feat device info: {}'.format(feat.device))
        # print('label device info: {}'.format(label.device))
        device = feat.device 
        loss =  torch.Tensor(0).to(device)
        batch_size = feat.size(0)
        for i in range(batch_size):
            feat_s = feat[i].view(-1, self.feat_dim)
            label_s = label[i].view(-1)
            if feat_s.size(1) != self.feat_dim:
                raise ValueError("Center's dim: {0} should be equal to input feature's \
                                dim: {1}".format(self.feat_dim,feat_s.size(1)))
            
            batch_size_tensor = feat_s.new_empty(1).fill_(feat_s.size(0) if self.size_average else 1)
            loss = self.centerlossfunc(feat_s, label_s, self.centers, batch_size_tensor)
            loss+=loss
        return loss/batch_size


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

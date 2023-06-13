from torchvision.transforms.functional import normalize
import torch.nn as nn
import numpy as np
import os 

def denormalize(tensor, mean, std):
    mean = np.array(mean)
    std = np.array(std)

    _mean = -mean/std
    _std = 1/std
    return normalize(tensor, _mean, _std)

class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def split_body_and_edge(img, gt_mask, bin_threshold=None, threshold=None):

    '''
      分离签名主体及边缘歧义区域
      img: 0~255
      gt: 0~255
    '''
    # 1. 
    _, gt_mask_thre = cv2.threshold(gt_mask,bin_threshold,255, cv2.THRESH_BINARY)
    sig_roi = np.multiply(img, (gt_mask_thre//255)[...,np.newaxis])
    if len(img)==3:
        sig_roi_gray = cv2.cvtColor(sig_roi, cv2.COLOR_BGR2GRAY) 
    else:
        sig_roi_gray = img 

    # 获取激活图
    # sigmoid_map  = 1/(1 + np.exp(sig_roi_gray*-1))
    # sigmoid_map = sigmoid_map
    # print(np.unique(sigmoid_map))
    sigmoid_map = (255-sig_roi_gray)
    sigmoid_map = np.multiply(sigmoid_map, gt_mask_thre//255)

    # 获取签名主体
    _,sig_body = cv2.threshold((sigmoid_map).astype('uint8'), threshold,255,cv2.THRESH_BINARY)

    # 获取边缘歧义区域
    sig_body_blur = gaussianblur(sig_body, 5)
    _,sig_body_blur = cv2.threshold(sig_body_blur, threshold,255,  cv2.THRESH_BINARY)
    sig_edge = sig_body_blur - sig_body
    return sig_body, sig_edge



def make_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
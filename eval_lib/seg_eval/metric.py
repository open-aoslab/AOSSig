'''
Description: 
Version: 2.0
Author: shuanghao chen
Date: 2023-06-09 01:49:59
'''
import numpy as np 
from skimage.metrics import structural_similarity  
import copy 


################ compute F1 score #################
class iandu_iv(object):
    def __init__(self, 
                 max_index = None):
        self.max_index = max_index

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        x, gt = x.copy(), gt.copy()
        
        if self.max_index is not None:
            max_index = self.max_index
        else:
            max_index = max(np.max(x)+1, gt.shape[1])

        bs = x.shape[0]
        if m is None:
            m = np.ones(x.shape, dtype=np.uint8)
            
        if gt.shape[1] > max_index:
            mgt = m*(gt[:, max_index:].sum(axis=1)==0)
        else:
            mgt = m
        
        gt = gt[:, 0:max_index]
        gt *= mgt[:, np.newaxis]
        gt = np.concatenate([gt, (mgt==0)[:, np.newaxis]], axis=1)
        # the last channel is ignore and it is mutually exclusive
        # with other multi class.

        mx = mgt*((x>=0)&(x<max_index))
        x[mx==0] = max_index # here set it with a new index
        # any predictions with its corresponding 
        # gts == ignore, will be forces labeled ignore so they will be
        # excluded from the confusion matrix.

        bdd_index = max_index+1 # include the ignore

        i, u = [], []
        for bi in range(bs):
            ii, uu = self.iandu_iv_single(x[bi], gt[bi], bdd_index)
            i.append(ii)
            u.append(uu)
        i = np.stack(i)[:, :max_index] # remove ignore
        u = np.stack(u)[:, :max_index]
        return i, u

class iandu_auto(object):
    def __init__(self,
                 max_index):
        self.max_index = max_index

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        x_shape = list(x.shape)
        gt_shape = list(gt.shape)
        

        if x_shape == gt_shape:
            return iandu_normal(self.max_index)(x, gt, m)
        elif x_shape == gt_shape[0:1]+gt_shape[2:]:
            return iandu_iv(self.max_index)(x, gt, m)
        else:
            raise ValueError

class iandu_normal(object):
    def __init__(self, 
                 max_index=None):
        self.max_index = max_index # 1

    def __call__(self,
                 x,
                 gt,
                 m=None,
                 **kwargs):
        if x.dtype not in [np.bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError
        if gt.dtype not in [np.bool, np.uint8, np.int32, np.int64, int]:
            raise ValueError

        x, gt = x.copy().astype(int), gt.copy().astype(int)

        if self.max_index is None:
            max_index = max(np.max(x), np.max(gt))+1
        else:
            max_index = self.max_index
        
        bs = x.shape[0]
        if m is None:
            m = np.ones(x.shape, dtype=np.uint8) # 构建x大小的单位矩阵

        mgt = m*(gt>=0)&(gt<max_index) # 有效区域
        gt[mgt==0] = max_index 
        # here set it with a new "ignore" index

        mx = mgt*((x>=0)&(x<max_index))
        # all gt ignores will be ignored, but if x predict
        # ignore on non-ignore pixel, that will count as an error.
        x[mx==0] = max_index 

        bdd_index = max_index+1 
        # include the ignore
            
        i, u = [], [] 

        for bi in range(bs):
            cmp = np.bincount((x[bi]+gt[bi]*bdd_index).flatten())
            cm = np.zeros((bdd_index*bdd_index)).astype(int) # 3x3
            cm[0:len(cmp)] = cmp
            cm = np.reshape(cm, (bdd_index, bdd_index))
            pdn = np.sum(cm, axis=0)
            gtn = np.sum(cm, axis=1)
            tp = np.diag(cm)
            i.append(tp)
            u.append(pdn+gtn-tp)            
        i = np.stack(i)[:, :max_index] # remove ignore
        u = np.stack(u)[:, :max_index]
        return i, u

class label_count(object):
    def __init__(self, 
                 max_index=None):
        self.max_index = max_index

    def __call__(self,
                 x,
                 m=None,
                 **kwargs):
        if x.dtype not in [np.bool, np.uint8, 
                           np.int32, np.uint32, 
                           np.int64, np.uint64,
                           int]:
            raise ValueError

        x = x.copy().astype(int)
        if self.max_index is None:
            max_index = np.max(x)+1
        else:
            max_index = self.max_index
        
        if m is None:
            m = np.ones(x.shape, dtype=int)
        else:
            m = (m>0).astype(int)

        mx = m*(x >=0)&(x <max_index)
        # here set it with a new "ignore" index
        x[mx==0] = max_index 

        count = []
        for bi in range(x.shape[0]):
            counti = np.bincount(x[bi].flatten())
            counti = counti[0:max_index]
            counti = list(counti)
            if len(counti) < max_index:
                counti += [0] * (max_index - len(counti))
            count.append(counti)
        count = np.array(count, dtype=int)
        return count

def compute_fscore(p,g,class_n, m=None, classname=None):
    '''
    inputs:
       p:pred [b, h, w]
       g:gt [b, h, w]
       class_n： class number
       classname: class name 
    return：dict
    '''

    if classname is None:
        classname = [
            'backgound',
            'text'
        ]

    tp, union = iandu_auto(class_n)(p, g, m)
    p_modified = copy.deepcopy(p) 
    p_modified[g>=class_n] = class_n 
    pn = label_count(class_n)(p_modified)
    gn = label_count(class_n)(g)

    pn_save = copy.deepcopy(pn); pn_save[pn==0]=1
    gn_save = copy.deepcopy(gn); gn_save[gn==0]=1
    prec_imwise = tp.astype(float)/(pn_save.astype(float))
    recl_imwise = tp.astype(float)/(gn_save.astype(float))
    prec_imwise[pn==0] = 0
    recl_imwise[gn==0] = 0

    prec_imwise = prec_imwise.mean(axis=0)
    recl_imwise = recl_imwise.mean(axis=0)
    fscore_imwise = 2*prec_imwise*recl_imwise/(prec_imwise+recl_imwise+0.00008)

    class_num = len(fscore_imwise)
    if classname is None:
        cname_display = [
            str(i).zfill(3) for i in range(class_num)]
    else:
        cname_display = [
            str(i).zfill(3)+'-'+classname[i] for i in range(class_num)]

    return {
        cname_display[i]:fscore_imwise[i] for i in range(class_num)}
  

################ compute iou  #################
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def compute_iou(label_trues, label_preds, n_class):
    '''
    description: 
    param {*}
    return {*}
    '''    
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "iu":iu
    }

def compoute_iou_batch(pred, gt, class_n=2):
    '''
    inputs: 
       pred: [n,h,w]
       gt: [n,h,w]
       class_n: the class number of dataset
    return: list 
    '''
    assert pred.shape == gt.shape, 'the size nconsistent of pred and gt {}'
    miou, fiou, biou = [], [],[]
    b, h, w = pred.shape
    for p, g in zip(pred,gt):
        res = compute_iou(p, g, class_n)
        miou.append(res['Mean IoU'])
        fiou.append(res['iu'][1])
        biou.append(res['iu'][0])
    return [sum(miou)/b, sum(fiou)/b, sum(biou)/b]


################ compute SSIM   #################
def matte_SSIM(pred_matte, gt_matte):
    '''
    description: 计算结构相似性SSIM
    param {*}
    return {*}
    '''    
    SSIM = structural_similarity(pred_matte, gt_matte)
    return SSIM  

def handwriting_segmentation_eval(pred_mask, gt_matte, nums_class = 2):
    res = compoute_iou_batch(pred_mask,gt_matte, nums_class)
    f_iou = res[1]
    f_score = compute_fscore(pred_mask, gt_matte, nums_class)
    f_score = f_score['001-text']
    ssim = matte_SSIM(np.uint8(pred_mask[0]*255), np.uint8(gt_matte[0]*255)) 
    return f_iou, f_score, ssim
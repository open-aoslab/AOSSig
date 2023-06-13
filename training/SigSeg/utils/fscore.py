'''
Author: chenshuanghao
Date: 2023-02-16 16:33:11
LastEditTime: 2023-06-02 15:34:12
Description: Do not edit
'''

import numpy as np 
import copy  

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
       p:pred [b, h, w]
       g:gt [b, h, w]
       class_n： class number
    '''

    if classname is None:
        classname = [
            'backgound',
            'text'
        ]

    tp, union = iandu_auto(class_n)(p, g, m)
    p_modified = copy.deepcopy(p) # [1,128,256]
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
    # print(prec_imwise, recl_imwise)
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
  

def compute_fscore_body_edge(p, body, edge, class_n):
    '''
       p: range [0,1]
       body: range [0,1]
       edge: range [0,1]
    '''
    body_text = None
    body_bg = None 
    edge_text = None 
    edge_bg = None 

    import cv2 

    # 计算主体
    body_p = np.multiply(p,1-edge)
    # cv2.imwrite('body_p.png',np.uint8(body_p*255))
    # cv2.imwrite('body.png',np.uint8(body*255))
    # print(np.unique(body_p))
    # print(np.unique(body))
   
    res = compute_fscore(body_p[np.newaxis, ...], body[np.newaxis,...], class_n)
    body_text = res['001-text']
    body_bg = res['000-backgound']

    # 计算边缘
    edge_p = np.multiply(p, edge)
    res =compute_fscore(edge_p[np.newaxis ,...], edge[np.newaxis, ...], class_n)
    edge_text = res['001-text']
    edge_bg = res['000-backgound'] 

    return body_text, body_bg, edge_text, edge_bg 



def compute_fscore_new_qxh(p, g, body,edge, class_n):

    '''
    '''
    # 计算全局的精确率和召回率
    g_p, g_r, g_f = compute_fscore_qxh(p[np.newaxis, ...], g[np.newaxis, ...], class_n)
    g_p = g_p['001-text']
    g_r = g_r['001-text']
    g_f = g_f['001-text']

    # 计算主体的精确率和召回率
    body_p = np.multiply(p,1-edge)
    l_p, l_r, l_f = compute_fscore_qxh(body_p[np.newaxis, ...], body[np.newaxis, ...], class_n) 
    l_p = l_p['001-text']
    l_r = l_r['001-text']
    l_f = l_f['001-text']
    # 计算总的F1 socre  
    f1_score_new =  2*g_p*l_r/(g_p+l_r)
    return  f1_score_new, g_f


def compute_fscore_new_qxh(p, g, body,edge, class_n):

    '''
    '''
    # 计算全局的精确率和召回率
    g_p, g_r, g_f = compute_fscore_qxh(p[np.newaxis, ...], g[np.newaxis, ...], class_n)
    g_p = g_p['001-text']
    g_r = g_r['001-text']
    g_f = g_f['001-text']

    # 计算主体的精确率和召回率
    body_p = np.multiply(p,1-edge)
    l_p, l_r, l_f = compute_fscore_qxh(body_p[np.newaxis, ...], body[np.newaxis, ...], class_n) 
    l_p = l_p['001-text']
    l_r = l_r['001-text']
    l_f = l_f['001-text']
    # 计算总的F1 socre  
    f1_score_new =  2*g_p*l_r/(g_p+l_r)
    return  f1_score_new, g_f


def compute_fscore_qxh(p,g,class_n, m=None, classname=None):
    '''
       p:pred [b, h, w]
       g:gt [b, h, w]
       class_n： class number
    '''

    if classname is None:
        classname = [
            'backgound',
            'text'
        ]

    tp, union = iandu_auto(class_n)(p, g, m)
    p_modified = copy.deepcopy(p) # [1,128,256]
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
    # print(prec_imwise, recl_imwise)
    fscore_imwise = 2*prec_imwise*recl_imwise/(prec_imwise+recl_imwise)

    class_num = len(fscore_imwise)
    if classname is None:
        cname_display = [
            str(i).zfill(3) for i in range(class_num)]
    else:
        cname_display = [
            str(i).zfill(3)+'-'+classname[i] for i in range(class_num)]

    return [{cname_display[i]:prec_imwise[i] for i in range(class_num)},\
            {cname_display[i]:recl_imwise[i] for i in range(class_num)},\
            {cname_display[i]:fscore_imwise[i] for i in range(class_num)},
            ]








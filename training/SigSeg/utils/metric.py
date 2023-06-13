import numpy as np
# from skimage.measure import compare_psnr, compare_ssim
import cv2  
import scipy   
from skimage.metrics import structural_similarity  
from skimage.measure import label


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist



def compoute_iou(pred, gt, class_n):
    '''
       pred: [n,h,w]
       gt: [n,h,w]
    '''
    assert pred.shape == gt.shape, 'the size nconsistent of pred and gt {}'
    miou, fiou, biou = [], [],[]
    b, h, w = pred.shape
    for p, g in zip(pred,gt):
        res = scores(p, g, class_n)
        miou.append(res['Mean IoU'])
        fiou.append(res['iu'][1])
        biou.append(res['iu'][0])
    return sum(miou)/b, sum(fiou)/b, sum(biou)/b



def compute_iou_body_edge(p, body, edge, class_n):
    '''
       p: range [0,1]
       body: range [0,1]
       edge: range [0,1]
    '''
    body_text = None
    body_bg = None 
    edge_text = None 
    edge_bg = None 


    # 计算主体
    body_p = np.multiply(p,1-edge)
    res = compoute_iou(body_p[np.newaxis, ...], body[np.newaxis, ...], class_n)
    body_text = res[1]
    body_bg = res[2]

    # 计算边缘
    edge_p = np.multiply(p, edge)
    res =compoute_iou(edge_p[np.newaxis, ...], edge[np.newaxis, ...], class_n)
    edge_text = res[1]
    edge_bg = res[2]

    return body_text, body_bg, edge_text, edge_bg 









def scores(label_trues, label_preds, n_class):
    '''
    description: 分割任务指标计算
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

def normlize(img):
    '''
    description: 图像归一化 
    param {*}
    return {*}
    '''    
    minValue = np.min(img)
    maxValue = np.max(img)

    if minValue>=0 and maxValue<=1:
        img = (img*255).astype(np.uint8)
        pass
    elif minValue>=-1 and minValue<0 and maxValue<=1:
        img = (((img+1)*0.5)*255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    return img 

def matte_SSIM(pred_matte, gt_matte):
    '''
    description: 计算结构相似性SSIM
    param {*}
    return {*}
    '''    
    SSIM = structural_similarity(pred_matte, gt_matte)
    return SSIM   


def gaussianblur(img, kesize=3):
    '''
        功能描述:
    '''
    gray = img.copy() 
    if len(img.shape)==3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kesize,kesize),0) 
    mask = np.zeros_like(gray)
    mask[gray>30] = 255
    return mask

def img_erode(img, k_size = 2):
    '''
     img: 为彩色输入 
    '''
    img_copy = img.copy() 
    kernel = np.ones((k_size, k_size),np.uint8)
    dst = cv2.erode(img_copy, kernel)
    unknow = img_copy - dst
    return dst, unknow 

def split_body_and_edge(img, gt_mask, bin_threshold=100, threshold=100):

    '''
      分离签名主体及边缘歧义区域
      img: 0~255
      gt: 0~255
    '''
    dst,unknow = img_erode(gt_mask)
    _, gt_mask_thre = cv2.threshold(gt_mask,bin_threshold,255, cv2.THRESH_BINARY)
    sig_roi = np.multiply(img, (gt_mask_thre//255))
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
    sig_body = dst+cv2.multiply(sig_body, unknow//255)

    # 获取边缘歧义区域
    sig_body_blur = gaussianblur(sig_body, 5)
    _,sig_body_blur = cv2.threshold(sig_body_blur, threshold,255,  cv2.THRESH_BINARY)
    sig_edge = sig_body_blur - sig_body
    return sig_body//255, sig_edge//255






#############################抠图指标：MAD,MSE,GRAD,CONN##################################
class MetricMAD:
    def __call__(self, pred, true):
        return np.abs(pred - true).mean() * 1e3

class MetricMSE:
    def __call__(self, pred, true):
        return ((pred - true) ** 2).mean() * 1e3

class MetricGRAD:
    def __init__(self, sigma=1.4):
        self.filter_x, self.filter_y = self.gauss_filter(sigma)
    
    def __call__(self, pred, true):
        pred_normed = np.zeros_like(pred)
        true_normed = np.zeros_like(true)
        cv2.normalize(pred, pred_normed, 1., 0., cv2.NORM_MINMAX)
        cv2.normalize(true, true_normed, 1., 0., cv2.NORM_MINMAX)

        true_grad = self.gauss_gradient(true_normed).astype(np.float32)
        pred_grad = self.gauss_gradient(pred_normed).astype(np.float32)

        grad_loss = ((true_grad - pred_grad) ** 2).sum()
        return grad_loss / 1000
    
    def gauss_gradient(self, img):
        img_filtered_x = cv2.filter2D(img, -1, self.filter_x, borderType=cv2.BORDER_REPLICATE)
        img_filtered_y = cv2.filter2D(img, -1, self.filter_y, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(img_filtered_x**2 + img_filtered_y**2)
    
    @staticmethod
    def gauss_filter(sigma, epsilon=1e-2):
        half_size = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
        size = np.int(2 * half_size + 1)

        # create filter in x axis
        filter_x = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                filter_x[i, j] = MetricGRAD.gaussian(i - half_size, sigma) * MetricGRAD.dgaussian(
                    j - half_size, sigma)

        # normalize filter
        norm = np.sqrt((filter_x**2).sum())
        filter_x = filter_x / norm
        filter_y = np.transpose(filter_x)

        return filter_x, filter_y
        
    @staticmethod
    def gaussian(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def dgaussian(x, sigma):
        return -x * MetricGRAD.gaussian(x, sigma) / sigma**2

class MetricCONN:
    def __call__(self, pred, true):
        step=0.1
        thresh_steps = np.arange(0, 1 + step, step)
        round_down_map = -np.ones_like(true)
        for i in range(1, len(thresh_steps)):
            true_thresh = true >= thresh_steps[i]
            pred_thresh = pred >= thresh_steps[i]
            intersection = (true_thresh & pred_thresh).astype(np.uint8)

            # connected components
            _, output, stats, _ = cv2.connectedComponentsWithStats(
                intersection, connectivity=4)
            # start from 1 in dim 0 to exclude background
            size = stats[1:, -1]

            # largest connected component of the intersection
            omega = np.zeros_like(true)
            if len(size) != 0:
                max_id = np.argmax(size)
                # plus one to include background
                omega[output == max_id + 1] = 1

            mask = (round_down_map == -1) & (omega == 0)
            round_down_map[mask] = thresh_steps[i - 1]
        round_down_map[round_down_map == -1] = 1

        true_diff = true - round_down_map
        pred_diff = pred - round_down_map
        # only calculate difference larger than or equal to 0.15
        true_phi = 1 - true_diff * (true_diff >= 0.15)
        pred_phi = 1 - pred_diff * (pred_diff >= 0.15)

        connectivity_error = np.sum(np.abs(true_phi - pred_phi))
        return connectivity_error / 1000

class MetricDTSSD:
    def __call__(self, pred_t, pred_tm1, true_t, true_tm1):
        dtSSD = ((pred_t - pred_tm1) - (true_t - true_tm1)) ** 2
        dtSSD = np.sum(dtSSD) / true_t.size
        dtSSD = np.sqrt(dtSSD)
        return dtSSD * 1e2


def alpha_evaluation(pred, target):
    '''
    description: 抠图结果评估
    param {*}
    return {*}
    '''    
    pred = pred.unsqueeze(1).cpu().detach().numpy()
    target = target.cpu().detach().numpy() 
    assert pred.shape == target.shape, 'the inconsistent of pred and target'
    eval_type = ['mse', 'mad', 'conn', 'grad']
    metrics  = { t:[] for t in eval_type}

    mad = MetricMAD()
    mse = MetricMSE()
    grad = MetricGRAD()
    conn = MetricCONN()
    b_size = pred.shape[0]
  
    for p, t in zip(pred, target):
        metrics['mad'].append(mad(p,t)) 
        metrics['mse'].append(mse(p,t)) 
        metrics['conn'].append(conn(p,t)) 
        metrics['grad'].append(grad(p,t)) 
    
    return {'mad':metrics['mad']/b_size, \
            'mse':metrics['mse']/b_size, \
            'conn':metrics['conn']/b_size, \
            'grad':metrics['grad']/b_size}
    
































# if __name__ == '__main__':
#     img_path = 'temp.png'
#     img_path2 = 'temp_test.png'
#     img = cv.imread(img_path, 0)
#     img2 = cv.imread(img_path2, 0)

#     out = matte_SSIM(img, img2)
#     print(out)

      


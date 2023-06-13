import sys 
sys.path.append('./././')
from eval_lib.seg_eval.metric import handwriting_segmentation_eval
import argparse
import os 
from PIL import Image
import numpy as np   
import cv2 


def get_args():
    parser = argparse.ArgumentParser(description='Handwriting segmentation task evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gt_root', type=str, required=True, default=None, help='the directory gt mask' ) 
    parser.add_argument('-p', '--pred_root', type=str, required=True, default=None, help='the directory pred mask' ) 
    return parser.parse_args()


def main():
    '''
        the handwriting segmentation eval 
    '''
    parser = get_args()  
    gt_root = parser.gt_root
    pred_root = parser.pred_root 

    gt_names = [name for name in os.listdir(gt_root) if name.endswith('.png')]
    f_iou, f_score, mssim = [], [], []
    for gt_name in gt_names:
        gt_img = np.array(Image.open(os.path.join(gt_root, gt_name)).convert('L'))
        pred_img = np.array(Image.open(os.path.join(pred_root, gt_name)).convert('L'))
        assert gt_img.shape == pred_img.shape,'the size of gt_img and pred_img are inconsistent'

        gt_mask = np.zeros_like(gt_img)
        gt_mask[gt_img>0] = 1 
        pred_mask = np.zeros_like(pred_img)
        pred_mask[pred_img>0] = 1 

        iou, score, ssim = handwriting_segmentation_eval(pred_mask[np.newaxis,...], gt_mask[np.newaxis,...]) 
        f_iou.append(iou)
        f_score.append(score)
        mssim.append(ssim)
    res = {
        'sample nums': len(gt_names), 
        'f_iou': sum(f_iou)/len(f_iou),
        'f_score': sum(f_score)/len(f_score), 
        'mssim': sum(mssim)/len(mssim) 
    }
    return res


if __name__ == '__main__':
    res = main() 
    print(res)

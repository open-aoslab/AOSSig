import torch
import torch.utils.data as data
import os 
import cv2 
import numpy as np 
import torchvision.transforms as transforms
from PIL import Image




class ImageValDataset(data.Dataset):

    def __init__(self, gt_root, pred_root):
        self.gt_root = gt_root
        self.pred_root = pred_root

        self.gt_paths = [ os.path.join(gt_root, name) for name in os.listdir(gt_root) if  name.endswith('.txt')]
        
    def __len__(self):
        return len(self.gt_paths)


    def __getitem__(self, index):

        data = {}
        gt_name = os.path.basename(self.gt_paths[index]) 
       
        gt_path = os.path.join(self.gt_root, gt_name.split('.')[0]+'.txt')
        pred_path = os.path.join(self.pred_root, gt_name.split('.')[0]+'.txt')
        

        # get gt 
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() 
        polys = []
        for line in lines:
            parts = line.strip().split(',')
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            poly = np.array(list(map(float, line[:8]))).reshape((-1, 2))
            polys.append(poly)

        # get pred 
        with open(pred_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() 
        pred_polys = []
        scores = []
        for line in lines:
            parts = line.strip().split(',')
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in parts]
            pred_poly = np.array(list(map(float, line[:8]))).reshape((-1, 2))
            if len(line)==9:
                box_score = float(line[-1])
            else:
                box_score=1.0 
            scores.append(box_score)
            pred_polys.append(pred_poly)

            
        ignore_tags = np.zeros((len(polys)))
        # data['image'] = self.transform(Image.fromarray(img))
        # data['shape'] = torch.from_numpy(np.array(img.shape[0:2]))
        data['polygons'] = torch.from_numpy(np.array(polys))
        data['filename'] = gt_name
        data['ignore_tags'] = torch.from_numpy(ignore_tags)
        data['box_pred'] = torch.from_numpy(np.array(pred_polys))
        data['score_pred'] = torch.from_numpy(np.array(scores))
        return data 
 
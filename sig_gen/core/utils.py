
import numpy as np  
from PIL import Image
import cv2   
import json 


def get_image(img_path, is_gray=False):
    if is_gray:
        img = np.array(Image.open(img_path).convert('L'))
    else:
        img = np.array(Image.open(img_path).convert('RGB'))
    return img 

def write_img(img, img_save_path):
    img = Image.fromarray(img.astype('uint8'))
    img.save(img_save_path)

def remove_blank(image,mask, offset=3):
    '''
    description: 移除掩膜图像中的背景空白 
    param {*}
    return {*}： 
    ''' 
    image = cv2.copyMakeBorder(image,3,3,3,3, cv2.BORDER_CONSTANT, value=(0,0,0))       
    mask = cv2.copyMakeBorder(mask,3,3,3,3, cv2.BORDER_CONSTANT, value=(0))  
    
    if len(mask.shape)>2:
        gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray_img= mask.copy()  
        
    ret,thresh1 = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
    h, w = gray_img.shape
    y_pos, x_pos = np.where(thresh1==255)
    min_y, max_y = np.min(y_pos), np.max(y_pos)
    min_x, max_x = np.min(x_pos), np.max(x_pos)
    # 边界周围轻微扩充
    if offset>0:
        min_y =  min_y-offset if min_y-offset>0 else min_y
        max_y =  max_y+offset if max_y+offset<h else max_y
        max_x =  max_x+offset if max_x+offset<w else max_x
        min_x=  min_x-offset if min_x-offset>0 else min_x
    
    return image[min_y:max_y, min_x:max_x], mask[min_y:max_y, min_x:max_x]

def resize_img_keep_ratio(sig_img,sig_mask,target_size):
    '''
    '''
    old_size= sig_img.shape[0:2]
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    sig_img = cv2.resize(sig_img,(new_size[1], new_size[0]))
    sig_mask = cv2.resize(sig_mask,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    sig_img_new = cv2.copyMakeBorder(sig_img,top,bottom,left,right,cv2.BORDER_REPLICATE)
    sig_mask_new = cv2.copyMakeBorder(sig_mask,top,bottom,left,right,cv2.BORDER_CONSTANT, None, (0))
    return sig_img_new, sig_mask_new

def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

def check_is_overlapping(region, boxes, threshold=0.1):
    '''
        判断是否存在候选框与其它候选框重叠
        flag： False, not overlapping True: overlapping
    '''
    flag = False
    for box in boxes:
        if cal_iou(region, box)>threshold:
            flag = True
            break
    return flag

def decode_json(json_path):
    '''
    labelme json位置信息解析,根据标签类别，返回矩形框信息
    '''
    sig_boxes  = []
    bg_boxes  = []
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    objs = data['shapes']
    for obj in objs:
        points = obj['points']
        label = obj['label']
        x1 = int(points[0][0])
        y1 = int(points[0][1])
        x3 = int(points[1][0])
        y3 = int(points[1][1])
        if label == 'sig':
            sig_boxes.append([x1,y1, x3, y3])
        elif label == 'bg':
            bg_boxes.append([x1,y1, x3, y3])
    return sig_boxes, bg_boxes


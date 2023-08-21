'''
Author: chenshuanghao
Date: 2023-04-07 00:24:13
LastEditTime: 2023-06-07 18:36:22
Description: Do not edit
'''


'''
Author: chenshuanghao
Date: 2023-04-04 19:57:40
LastEditTime: 2023-04-05 17:24:50
Description: Do not edit
'''
from locale import normalize
import cv2 
import numpy as np
from PIL import Image  
import random 
from tools import poisson
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from text_renderer.effect.imgaug_effect import Emboss, MotionBlur,PerspectiveTransform
import os
import json 




sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential(
    [
        sometimes(iaa.Affine(rotate = (-10, 10))),
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                iaa.Sharpen(alpha=(0, 0.75), lightness=(0.75, 1.0)), # sharpen images
                # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
            
                # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                # iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True
)


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



   
def sig_mask_aug(sig_img, sig_mask):
    '''
       签名及签名mask数据增广
       sig_mask: [0-255]
    '''
    segmap = SegmentationMapsOnImage(np.uint8(sig_mask), shape=sig_mask.shape)
    sig_img, sig_mask = seq(image =sig_img, segmentation_maps=segmap)
    sig_mask = np.uint8(sig_mask.get_arr())
    return sig_img, sig_mask  
  


def compute_sig_postion2(img_size,sig_size):
    '''
       功能描述： 给出签名局部位置以及签名，计算合适的候选区域
    '''
       
    H,W = img_size
    h,w = sig_size
    # 相较于局部签名候选区域
    random_x = random.randint(0, W-w-1)
    random_y = random.randint(0, H-h-1)
    return [random_x, random_y, random_x+w, random_y+h]
    

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
        判断候选框与其它候选框是否存在重叠
        flag： False, not overlapping True: overlapping
    '''
    flag =False
    for box in boxes:
        if cal_iou(region, box)>threshold:
            flag = True
            break
    return flag






def get_text_color(bg: Image):
    # TODO: better get text color
    np_img = np.array(bg)
    mean = np.mean(np_img)
    alpha = np.random.randint(190, 255)
    r = np.random.randint(0, int(mean * 0.7))
    g = np.random.randint(0, int(mean * 0.7))
    b = np.random.randint(0, int(mean * 0.7))
    fg_text_color = (r, g, b, alpha)
    return fg_text_color  


def paste_image(bg_img, sig_mask):

    bg_img = Image.fromarray(bg_img)
    sig_mask = Image.fromarray(sig_mask)
    sig_mask,_ = PerspectiveTransform()(sig_mask, None)

    # 1. text_color
    bg_img = bg_img.convert('RGBA')
    sig_color = get_text_color(bg_img) # 替换为红色 
    

    # 随机透视变换
    mask_ = np.array(sig_mask.convert('L'))
    mask = np.zeros_like(mask_)
    mask[mask_>50] = 1
    
    # 2. sig_mask convert
    sig_mask = sig_mask.convert('RGBA')
    sig_mask = np.array(sig_mask)
    sig_mask[:,:,3]=np.uint8(mask*255)
    if random.random()>0.6:
        sig_mask[mask>0,:] = sig_color
    else:
        sig_mask[mask>0,:] = None 


    sig_mask = Image.fromarray(sig_mask)

    # 3. resize 大小调整
    bg_img.paste(sig_mask,(0,0),mask=sig_mask.split()[-1])
    # 4. 图像粘贴
    bg_img = bg_img.convert('RGB')
    bg_img = np.array(bg_img)
    sig_local = bg_img*mask[...,np.newaxis]
    sig_local = cv2.cvtColor(sig_local, cv2.COLOR_RGB2GRAY)
    sig_local = np.concatenate([sig_local[...,np.newaxis]]*3, 2)
    bg_img = (1-mask[...,np.newaxis])*bg_img+sig_local
    
    if random.random()>1.0:
        if random.random()>0.5:
            # sig_img[sig_mask>200,:] = [197, 144 ,123] # 蓝色 B,G,R
            bg_img[mask>0,:] = [123, 144 ,197] # 蓝色 B,G,R
        else:
            # sig_img[sig_mask>200,:] = [127, 111, 182] # 红色
            bg_img[mask>0,:] = [182, 111, 127] # 红色

    return np.uint8(bg_img), np.uint8(mask*255)
    



def merge_image(source_img, target_img, mask_img):
    '''
        泊松图像融合
        source_img: 前景图片 
        target_img: 背景图片
        mask_img：ROI区域
    '''
    _,mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    assert len(np.unique(mask_img)) == 2 , 'value:{}'.format(','.join([str(i) for i in np.unique(mask_img)]))
    if source_img.shape[0:2]!=target_img.shape[0:2]:
        h, w = source_img.shape[0:2]
        target_img = cv2.resize(target_img,(w,h), cv2.INTER_AREA)
        
    # Normalize mask to range [0,1]
    # mask = np.atleast_3d(mask_img).astype(np.float) / 255.
    mask = mask_img/255.
    h,w = mask.shape 
    assert np.sum(mask)<h*w//2
    # Make mask binary
    # mask[mask_img != 1] = 0
    # Trim to one channel
    # mask = mask[:,:,0]
    # cv2.imwrite('test.png', np.uint8(mask*255))
    channels = source_img.shape[-1]
    # Call the poisson method on each individual channel
    result_stack = [poisson.process(source_img[:,:,i], target_img[:,:,i], mask) for i in range(channels)]
    # Merge the channels back into one image
    result_new = []
    for re in result_stack:
        re[re<0] = 0 
        re[re>255] = 255
        result_new.append(re)
    result_new =cv2.merge(result_new)
    

    return result_new, mask_img 


def normlize(img, is_rgb=True):
    min_x = np.min(img,axis=(0,1))
    max_x = np.max(img,axis=(0,1))
    res = (img-min_x)/(max_x-min_x)
    return np.uint8(res*255)

def merge_image_opencv(obj, dst, mask):
    '''
       泊松融合opencv api实现
    '''

    height, width, channels= dst.shape
    center =  (width//2, height//2)
    # mask = 255 * np.ones(obj.shape, obj.dtype)
    # mask = np.concatenate([mask[...,np.newaxis]]*3, 2)
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 5)


    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)
    # mixed_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)
    
    # for i in range(3):
    #     normal_clone[normal_clone[:,:,i]<0, i] = 0
    #     normal_clone[normal_clone[:,:,i]>255, i] = 255
    # normal_clone = normlize(normal_clone)
    return normal_clone, mask 

def resize_img_keep_ratio(sig_img,sig_mask,target_size):
    old_size= sig_img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
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

def merge_image_violence(obj, dst, mask):
    '''
       直接暴力对两张图片进行粘和
       input: obj  前景
              dst  背景
              mask  签名mask
    '''
    mask_inverse = ~mask
    dst = (dst*(mask_inverse/255.0)[...,np.newaxis]).astype('uint8')
    obj = (obj*(mask/255.0)[...,np.newaxis]).astype('uint8')

    return np.uint8(dst+obj),mask

def get_image(img_path, is_gray=False):
  
    if is_gray:
        # img =cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        img = np.array(Image.open(img_path).convert('L'))
    else:
        # img =cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR)
        img = np.array(Image.open(img_path).convert('RGB'))
    
    return img 

def write_img(img, img_path):
    # cv2.imencode('.png',img)[1].tofile(img_path)
    img = Image.fromarray(img.astype('uint8'))
    img.save(img_path)

def remove_blank(image,mask):
    '''
    description: 移除背景空白 
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
    # cv2.imwrite('threshold.png', thresh1)
    h, w = gray_img.shape
    y_pos, x_pos = np.where(thresh1==255)
    min_y, max_y = np.min(y_pos), np.max(y_pos)
    min_x, max_x = np.min(x_pos), np.max(x_pos)
    # 边界自动扩充
    offset = 3
    min_y =  min_y-offset if min_y-offset>0 else min_y
    max_y =  max_y+offset if max_y+offset<h else max_y
    max_x =  max_x+offset if max_x+offset<w else max_x
    min_x=  min_x-offset if min_x-offset>0 else min_x
    return image[min_y:max_y, min_x:max_x], mask[min_y:max_y, min_x:max_x]

def synthesis_sig(img_local, sig, mask):
        '''
        description: 在背景随机位置进行签名合成
        param {*}
        return {*}： 合成后背景图像、 对应的签名mask 
        '''        
    
        img = img_local.copy()  
        sig_img = sig
        sig_mask = np.uint8(mask>0)

        if len(sig_img.shape)==2:
            sig_img = np.concatenate([sig_img[...,np.newaxis]]*3, 2)
        sig_img = sig_img*sig_mask[...,np.newaxis]

        region = compute_sig_postion2(img.shape[0:2], sig_img.shape[0:2])
        x1, y1, x3, y3 = region
        
        # 获取背景块
        bg_block = img[y1:y3, x1:x3, :]
        bg_block_copy = bg_block.copy()
        # 获取前景块
        # foreground = np.concatenate([sig_img[...,np.newaxis]]*3,-1)
        foreground = sig_img
        foreground = foreground.astype(float)
        bg_block = bg_block.astype(float)
        alpha = random.uniform(0.85, 1.0)

        foreground = alpha * foreground
        bg_block = (1.0 - alpha) * bg_block
        fusion_img = cv2.add(foreground, bg_block)
        only_sig_region = fusion_img*sig_mask[...,np.newaxis]
        only_bg_region = (1-sig_mask)[...,np.newaxis]*bg_block_copy
        img[y1:y3, x1:x3, :] = only_sig_region+only_bg_region

        mask_new = np.zeros((img.shape[0:2]))
        mask_new[y1:y3, x1:x3] = sig_mask*255
        return img, mask_new

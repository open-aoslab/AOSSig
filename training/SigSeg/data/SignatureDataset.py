import cv2 as cv
import os
from PIL import Image
from torch.utils import data
import torch
from torchvision.transforms.transforms import Pad
import torchvision.transforms as transforms
from data.data_aug_tools import Random_image_blur, Random_image_color_jitter, \
    Random_image_hflip, Random_image_vflip, Resize, Compose, ToTensor, Normalize
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random
import numpy as np 


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-15, 15), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                # iaa.Invert(0.05, per_channel=True), # invert color channels
                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply((0.5, 1.5), per_channel=True),
                        second=iaa.LinearContrast((0.5, 2.0))
                    )
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                
                # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        ),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        
    ],
    random_order=True
)


class SignatureDataset(data.Dataset):

    def __init__(self, dataroot=None,\
              mode='train', \
              data_dir = 'data_local', \
              mask_dir='mask_local' , \
              transform=None):
        super(SignatureDataset,self).__init__()
        self.dataroot = dataroot
        self.mode = mode
        self.data_dir = os.path.join(self.dataroot,self.mode,data_dir)
        self.binary_thersold = 50

        if self.mode in ['train', 'val']:
            self.mask_dir = os.path.join(self.dataroot,self.mode,mask_dir)
        self.imgs = [name for name in os.listdir(self.data_dir) if '.png' in name or '.jpg' in name]

        if self.mode == 'test':
            self.transform = transforms.Compose([
               transforms.Resize((128, 256)), 
               transforms.ToTensor(),
               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ]
            ) 
        else:
            self.transfrom = Compose([
                Resize(), 
                Random_image_hflip(),
                Random_image_vflip(),
                # Random_image_blur(),
                # Random_image_color_jitter(),
                ToTensor(),
                Normalize(is_gray=False)
            ])        
        
    def __getitem__(self,index):

        label_dict = {}
        sig_path =os.path.join(self.data_dir, self.imgs[index])
        sig = Image.open(sig_path).convert('RGB')
        
        if self.mode == 'test':
            input_tensor = self.transform(sig)   
            return {
                'image':input_tensor,
                'img_name':self.imgs[index]
            }  
        else:
            prefix = self.imgs[index].split('.png')[0]
            mask_path = os.path.join(self.mask_dir, prefix+'.png')
            mask_ = get_image(mask_path,True, norm=False, expand=False, resize=False)
            mask = np.zeros_like(mask_)
            mask[mask_>self.binary_thersold]=1
            mask = Image.fromarray(mask)
            img, mask = self.transfrom(sig, mask)
            label_dict['image'] = img
            label_dict['mask'] = mask.long() 
            return label_dict 

    def __len__(self):
        return len(self.imgs)



def get_image(img_path, gray_convert=False, norm=False, expand= True, resize=True):
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8),1)

    if gray_convert:
       img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if resize:
        img = cv.resize(img, (512, 512))

    if norm:
        img = img/255.0

    if expand:
        img = img[..., np.newaxis]

    return img






import augly.image as imaugs
import PIL.Image as Image
import torchvision.transforms as transform
import random
import numpy as np  
import torch  


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size 
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ToTensor(object):
    def __init__(self): 
        self.transform = transform.ToTensor()

    def __call__(self, img, mask):
        img = self.transform(img)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return img, mask

class Normalize(object):
    def __init__(self, is_gray=False): 
        self.normal_transform = transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        if is_gray:
            self.normal_transform = transform.Normalize((0.5),(0.5))

    def __call__(self, img, mask):
        img = self.normal_transform(img)
        return img, mask
    
class Random_image_blur(object):
    
    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
        
        assert img.size == mask.size, 'blur'
        
        if random.random()>0.7:
            radius = random.uniform(0.2, 2)
            img = imaugs.blur(img, radius=radius)
        
        assert img.size == mask.size, 'blur'
        return img, mask

class Random_image_brightness(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
     
        if random.random()>0.7:
            factor = random.uniform(0.6, 1.5)
            img = imaugs.brightness(img, factor=factor)
        return img, mask

class Random_image_color_jitter(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
        
        assert img.size == mask.size, 'color'
        if random.random()>0.7:
            brightness_factor  = random.uniform(0.5, 1.5)
            contrast_factor = random.uniform(0.5, 1.5)
            factor = random.uniform(0.5, 1.5)
            img = imaugs.color_jitter(img,brightness_factor=brightness_factor,contrast_factor=contrast_factor,saturation_factor=factor)
        
        assert img.size == mask.size, 'color'
        return img, mask

class Random_image_stripes(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask, trimap):

        if random.random()>0.5:
            line_angle = random.randint(-360,360)
            line_opacity = random.random()
            line_opacity = 0.6 if line_opacity<0.6 else line_opacity
            img = imaugs.overlay_stripes(img,line_width=0.05, line_color=(0, 0, 0),line_angle=line_angle,line_opacity=line_opacity)
        return img, mask, trimap

class Random_perspective_transform(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):

        if random.random()>0.0:
            sigma = random.randint(0, 30)
    
            img = imaugs.perspective_transform(img, sigma= sigma)
            mask = imaugs.perspective_transform(mask, sigma=sigma)  # 注意mask是通过插值算法来进行计算的,因此还需要重新获取相应的mask 
        return img, mask

class Resize(object):

    def __init__(self):
        pass  

    def __call__(self,img, mask):
        img = img.resize((256, 128),Image.ANTIALIAS) 
        mask = mask.resize((256, 128),Image.ANTIALIAS) 
        return img, mask  

class Random_add_noise(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
        if random.random()>0.8:
           img = imaugs.random_noise(img)

        return img, mask

class Random_image_rotate(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
        if random.random()>0.5:
           degree = random.randint(0,120)
           img = imaugs.rotate(img, degrees=degree)
           mask = imaugs.rotate(mask, degrees=degree)  

        return img, mask 

class Random_image_vflip(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
       
        assert img.size == mask.size, 'vflip'
        if random.random()>0.5:
           img = imaugs.vflip(img)
           mask = imaugs.vflip(mask)  
         
        
        assert img.size == mask.size, 'vflip'
        return img, mask

class Random_image_hflip(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):
        
        assert img.size == mask.size, 'hflip'
        if random.random()>0.5:
           img = imaugs.hflip(img)
           mask = imaugs.hflip(mask)  
        assert img.size == mask.size, 'hflip'
        return img, mask

class Random_shuffle_pixels(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask):

        if random.random()>0.5:
            ratio  = random.uniform(0.0, 0.2)
            img = imaugs.shuffle_pixels(img, factor=ratio )
        return img, mask


class Random_image_pixelization(object):

    def __init__(self, output_path=None):
        self.output_path = output_path

    def __call__(self, img, mask, trimap):
        if random.random()>0.5:
           img = imaugs.pixelization(img, ratio=0.4)
        return img, mask ,trimap





         

     

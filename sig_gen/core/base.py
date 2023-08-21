from re import S
import yaml 
from os.path import join as osp
import os 
import numpy as np   
import random  
from sig_gen.core import utils 
import cv2  
from PIL import Image 

class Base(object):
    def __init__(self) -> None:
        super().__init__()

        # 1. 加载并解析配置文件
        # 2. 加载元数据
        # 3. 启动相关默认配置

    def init_variables(self):

         # 合成素材
        self.bg_paths = [ osp(self.config['data']['background']['data_dir'],name) for name in os.listdir(self.config['data']['background']['data_dir']) if name.endswith('.jpg') or name.endswith('.png')][0:2]
        self.sig_paths = [ osp(self.config['data']['signature']['sig_img_dir'],name) for name in os.listdir(self.config['data']['signature']['sig_img_dir']) if name.endswith('.jpg') or name.endswith('.png')] 
      
        # 素材数据目录
        self.sig_mask_dir = self.config['data']['signature']['sig_mask_dir']
        self.res_anno_save_dir = self.config['data']['signature']['anno_save_dir']
        self.res_mask_save_dir = self.config['data']['signature']['mask_save_dir']
        self.res_img_save_dir = self.config['data']['background']['data_save_dir']
        self.sig_loc_dir = self.config['data']['signature']['sig_loc_dir']
        self.create_dir(self.res_anno_save_dir)
        self.create_dir(self.res_mask_save_dir)
        self.create_dir(self.res_img_save_dir)
        # 合成参数设置
        self.pos_mode = self.config['compose']['loc_label']['mode']
        self.pos_type = self.config['compose']['loc_label']['file_type']
        self.sig_select_mode = self.config['compose']['signature']['select_mode']
        self.sig_random_ratios = self.config['compose']['signature']['random_ratio']
        self.sig_max_nums = self.config['compose']['signature']['max_nums']
        self.ph_sh_ratio = self.config['compose']['signature']['ph_sh_ratio']
        self.pw_sw_ratio = self.config['compose']['signature']['pw_sw_ratio']
        self.sw_sh_ratio = self.config['compose']['signature']['sw_sh_ratio']
        self.iter_nums_per = self.config['compose']['iter_max_nums']
        # Text Render 
        self.fusion_type = self.config['compose']['fusion']
        self.sig_collect_path = self.config['compose']['text_render']['sig_collect_path']
        self.text_color_mode = self.config['compose']['text_render']['color_select']
        if self.fusion_type == 'text_render':
            self.get_samples_pixels()  
        
    def parse_yaml(self,yaml_path):
        '''
           配置文件参数解析 
        '''
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f.read(), Loader=yaml.FullLoader)
        return data

    def get_samples_pixels(self):
        '''
            加载文本像素集合
        '''
        pixels_array = np.load(self.sig_collect_path)
        self.pixes_mean = np.mean(pixels_array)
        self.pixels_array = pixels_array.tolist()
        
    def create_dir(self, data_root):
        if not os.path.exists(data_root):
            os.makedirs(data_root)

    def synthesis_core(self,img, img_name=None):
        '''
           单张背景下，签名数据合成 
        '''
        mask = np.zeros(img.shape[0:2])
        pos, p_label = self.compute_sig_postion(img.shape[0:2], img_name)
        if not len(pos):
            return None, None, None, None

        pos_res = []
        ocr_label = []
        for r in pos:
            sig_region = r
            sig_img, sig_mask , sig_name = self.random_get_signature(sig_region) 
            img_local = img[sig_region[1]:sig_region[3],sig_region[0]:sig_region[2],:]
            assert img_local.shape[0]>sig_img.shape[0] and img_local.shape[1]>sig_img.shape[1], 'off_w:{}, off_h:{}'.format(img_local.shape[0]-sig_img.shape[0], img_local.shape[1]-sig_img.shape[1])
    
            region = self.get_sig_postion_local(img_local.shape[0:2], sig_img.shape[0:2])
            x1, y1, x3, y3 = region
            sig_region_new = [sig_region[0]+x1, sig_region[1]+y1, sig_region[0]+x1+sig_img.shape[1],sig_region[1]+y1+sig_img.shape[0]]

           
            if self.fusion_type == 'possion':
                img_local_sig, sig_mask_local = self.merge_image_opencv(sig_img,img_local[y1:y3,x1:x3,:],sig_mask)
            elif self.fusion_type == 'violence':
                img_local_sig, sig_mask_local = self.merge_image_violence(sig_img,img_local[y1:y3,x1:x3,:],sig_mask)
            elif self.fusion_type == 'text_render':
                img_local_sig, sig_mask_local = self.paste_image(img_local[y1:y3,x1:x3,:],sig_mask)
            else:
                print(self.fusion_type+'is unsupported, only support the possion,violence, text_render')
                exit() 
        
            
            img_local[y1:y3,x1:x3,:] = img_local_sig
            mask_local = np.zeros(img_local.shape[0:2])
            mask_local[y1:y3,x1:x3] = sig_mask

            # if random.random()>1.0:
            #     # 获取印章、干扰所在区域
            #     h, w  = img.shape[0:2] 
            #     offset = 30
            #     x1,y1,x3,y3 = sig_region
            #     x1 = x1-offset if x1-offset>0 else x1 
            #     y1 = y1-offset if y1-offset>0 else y1  
            #     x3 = x3+offset if x3+offset>w else x3
            #     y3 = y3+offset if y3+offset>h else y3
            #     img_local_temp = img[y1:y3, x1:x3,:]
            #     img_local_sig = random_aug(img_local_temp)
            #     img[y1:y3, x1:x3,:] = img_local_sig

            # 4. 合成结果复原 
            img[sig_region[1]:sig_region[3],sig_region[0]:sig_region[2],:] = img_local
            mask[sig_region[1]:sig_region[3],sig_region[0]:sig_region[2]] = mask_local

            #5. add seal 
            if random.random()>1.0:
                img = self.add_seals(img, sig_region)
            pos_res.append(sig_region_new)
            ocr_label.append(sig_name)

        assert len(ocr_label)==len(p_label)
        return img, mask, pos_res, ocr_label, p_label
    
    def compute_sig_postion(self,img_size=None, img_name=None):
        '''
            获取签名粘贴位置
        '''
        if self.pos_mode=='random':
            # 功能描述: 基于签名与合同数据间的比例关系，自动选择一个签名区域
            self.get_sig_postion_random(img_size) 
 
        elif self.pos_mode == 'label':
            assert img_name is not None, 'the image name of json file is not None'
            json_path = osp(self.sig_loc_dir, img_name+'.json')
            if self.pos_type == 'json':
                sig_boxes,bg_boxes = utils.decode_json(json_path)
            elif self.pos_type == 'txt':
                sig_boxes, bg_boxes = None,None 
            else:
                sig_boxes, bg_boxes = None,None 
            labels = []
            total_boxes = []
            if self.sig_select_mode=='sig':
                total_boxes.extend(sig_boxes)
                labels.extend([ 'sig' for i in range(len(sig_boxes))])
            elif self.sig_select_mode=='bg':
                total_boxes.extend(bg_boxes)
                labels.extend([ 'bg' for i in range(len(bg_boxes))])
            elif self.sig_select_mode=='sig_bg':
                total_boxes.extend(sig_boxes)
                total_boxes.extend(bg_boxes)
                labels.extend([ 'sig' for i in range(len(sig_boxes))])
                labels.extend([ 'bg' for i in range(len(bg_boxes))])
            elif self.sig_select_mode == 'sig_bg_random':
                total_boxes.extend(sig_boxes)
                total_boxes.extend(bg_boxes)
                boxes_random_ = self.compute_sig_postion_random(img_size)
                # filter overlab 
                boxes_random = []
                for boxes in boxes_random_:
                    if not utils.check_is_overlapping(boxes, total_boxes):
                        boxes_random.append(boxes)
                total_boxes.extend(boxes_random)
                labels.extend([ 'sig' for i in range(len(sig_boxes))])
                labels.extend([ 'bg' for i in range(len(bg_boxes))])
                labels.extend([ 'random' for i in range(len(boxes_random))])
            
            # 随机采样特定个数的位置框
            r_index = random.randint(0, len(self.sig_random_ratios)-1)
            random_nums = int(self.sig_random_ratios[r_index]*len(total_boxes))
            random_boxe_indexs = random.sample(range(len(total_boxes)),  random_nums)
            total_boxes = np.array(total_boxes)
            labels = np.array(labels)
            random_boxes = total_boxes[random_boxe_indexs]
            random_boxe_labels = labels[random_boxe_indexs]
            return random_boxes, random_boxe_labels

    def update_ratio(self):
        H_h_ratios = [self.ph_sh_ratio-1, self.ph_sh_ratio, self.ph_sh_ratio+1]
        W_w_ratios = [self.pw_sw_ratio-1, self.pw_sw_ratio, self.pw_sw_ratio+1]
        ratios = [self.sw_sh_ratio-0.1, self.sw_sh_ratio, self.sw_sh_ratio+0.1]

        self.ph_sh_ratio = random.sample(H_h_ratios, 1)[0]
        self.pw_sw_ratio = random.sample(W_w_ratios, 1)[0]
        self.sw_sh_ratio = random.sample(ratios, 1)[0]

    def get_sig_postion_random(self, img_size):
        '''
            功能描述: 根据背景图像大小，随机生成不同大小范围的签名位置
        '''
        self.update_ratio() # 随机抖动签名相关宽高比
        pos = []
        H, W  = img_size
        w = int(W/self.pw_sw_ratio)
        h = int(H/self.ph_sh_ratio)
        max_times = 20
        while True:
            random_x = random.randint(0, W-w)
            random_y = random.randint(0, H-h)
            box = [random_x, random_y, random_x+w, random_y+h]
            if not utils.check_is_overlapping(box, pos):
                pos.append(box)
            if len(pos)==self.sig_max_nums:
                return pos 
            max_times-=1
            if not max_times:
                return pos 
        return pos  
     
    def random_get_signature(self, region, is_aug=False):
        '''
          随机选择签名
        '''
        x1,y1,x3,y3 = region 
        w, h = x3-x1, y3-y1
        ratio = round(w/h, 2)  

        index = random.randint(0, len(self.sig_paths)-1)
        sig_img = utils.get_image(self.sig_paths[index])
        sig_mask = utils.get_image(os.path.join(self.sig_mask_dir, os.path.basename(self.sig_paths[index]).replace('.jpg', '.png')), is_gray=True)
        sig_name = os.path.basename(self.sig_paths[index]).split('-')[0]
        
        if sig_img.shape[0:2]!=sig_mask.shape[0:2]:
            h_,w_ = sig_mask.shape[0:2]
            sig_img = cv2.resize(sig_img,(w_,h_))
        assert sig_img.shape[0:2] == sig_mask.shape[0:2]
        
        #2. 移除签名周围空白区域
        sig_img, sig_mask = utils.remove_blank(sig_img, sig_mask)
        s_h, s_w = sig_img.shape[0:2]
        # Resize至目标区域大小，并保持相应的宽高比 
        sig_img, sig_mask = utils.resize_img_keep_ratio(sig_img, sig_mask, (h, w))
       
        #3.签名随机尺度、等比例缩放Resize
        scale = random.choice([0.9, 0.95])
        sig_img = cv2.resize(sig_img,(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sig_mask = cv2.resize(sig_mask,(0,0), fx=scale, fy=scale, interpolation= cv2.INTER_NEAREST)
        return sig_img, sig_mask, sig_name

    def get_sig_postion_local(self,img_size,sig_size):
        '''
            给出签名局部位置以及签名，随机选择合适的候选区域
        '''
        H,W = img_size
        h,w = sig_size
        # 相较于局部签名候选区域
        random_x = random.randint(0, W-w-1)
        random_y = random.randint(0, H-h-1)
        return [random_x, random_y, random_x+w, random_y+h]
    
    def merge_image_opencv(self, obj, dst, mask):
        '''
          泊松融合opencv api实现
        '''
        height, width, channels= dst.shape
        center =  (width//2, height//2)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 5)
        normal_clone = cv2.seamlessClone(obj, dst, mask, center, cv2.MIXED_CLONE)
        return normal_clone, mask 
       
    def merge_image_violence(self, obj, dst, mask):
        '''
            直接暴力粘贴两张图片
        '''
        mask_inverse = ~mask
        dst = (dst*(mask_inverse/255.0)[...,np.newaxis]).astype('uint8')
        obj = (obj*(mask/255.0)[...,np.newaxis]).astype('uint8')
        return np.uint8(dst+obj),mask

    def get_text_color(self, bg: Image):
        '''
         获取文本采样颜色
        '''
        mean = self.pixes_mean
        alpha = np.random.randint(190, 255)
        r = np.random.randint(0, int(mean * 0.7))
        g = np.random.randint(0, int(mean * 0.7))
        b = np.random.randint(0, int(mean * 0.7))
        fg_text_color = (r, g, b, alpha)
        return fg_text_color

    def paste_image(self,bg_img, sig_mask):
        '''
           Text Renderer 合成代码
        '''
        bg_img = Image.fromarray(bg_img)
        sig_mask = Image.fromarray(sig_mask)
        
        # 1. text_color
        bg_img = bg_img.convert('RGBA')
        sig_color =self.get_text_color(bg_img) # 替换为红色 
        
        # 随机透视变换
        sig_mask_gray = np.array(sig_mask.convert('L'))
        sig_mask_bin = np.zeros_like(sig_mask_gray)
        sig_mask_bin[sig_mask_gray>50] = 1
        
        # 2. sig_mask padding 
        sig_mask = sig_mask.convert('RGBA')
        sig_mask = np.array(sig_mask)
        sig_mask[:,:,3]=np.uint8(sig_mask_bin*255)

        if self.text_color_mode =='fix':
            sig_mask[sig_mask_bin>0,:] = sig_color
        elif self.text_color_mode =='sample':
            sig_mask[sig_mask_bin>0,:] = self.sample_pixels(sig_mask_bin)[sig_mask_bin>0,:]
        else:
            pass
            # 混合采样

        sig_mask = Image.fromarray(sig_mask)
        # 3. resize 大小调整
        bg_img.paste(sig_mask,(0,0),mask=sig_mask.split()[-1])
        bg_img = np.array(bg_img.convert('RGB'))
        # 4. 图像粘贴
        # bg_img = bg_img.convert('RGB')
        # bg_img = np.array(bg_img)
        # sig_local = bg_img*mask[...,np.newaxis]
        # sig_local = cv2.cvtColor(sig_local, cv2.COLOR_RGB2GRAY)
        # sig_local = np.concatenate([sig_local[...,np.newaxis]]*3, 2)
        # bg_img = (1-mask[...,np.newaxis])*bg_img+sig_local
        
        return np.uint8(bg_img), np.uint8(sig_mask_bin*255)

    def sample_pixels(self, sig_mask):
        mask = sig_mask.copy()
        h, w = mask.shape[0:2]
        mask_sample = np.zeros((h, w, 4), dtype=np.uint8)
        indexs = np.argwhere(mask>0)
        for index in indexs:
            r_index = random.randint(0, len(self.pixels_array)-1)
            pixel = self.pixels_array[r_index].copy()
            pixel.append(255)
            y, x = index
            try:
                mask_sample[y][x] = pixel
            except Exception as e:
                print(pixel)
                print(mask_sample[y][x])
        return mask_sample

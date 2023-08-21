'''
Author: chenshuanghao
Date: 2023-06-07 23:56:14
LastEditTime: 2023-06-12 16:32:58
Description: Do not edit
'''
import sys 
sys.path.append('././')
from sig_gen.core import base
import os  
from sig_gen.core import utils 
import traceback 


class SigGenearation(base.Base):

    def __init__(self, config_path=None) -> None:
        super().__init__()
        self.config = super().parse_yaml(config_path)
        self.init_variables()

    def generation_main(self):
        '''
          手写签名检测、分割数据生成
        '''
        for bg_path in self.bg_paths:
            bg_name = os.path.basename(bg_path).split('.')[0]
            bg_img = utils.get_image(bg_path)
    
            for index in range(self.iter_nums_per):
                bg_img_copy = bg_img.copy()
                try:
                    img_gen, mask_gen, pos, ocr_label, p_label = self.synthesis_core(bg_img_copy, bg_name)
                except Exception as e:
                    traceback.print_exc(file=open('log.txt', 'a+'))
                    print(traceback.print_exc())

                if  img_gen is None or mask_gen is None : continue 
                # save path 
                bg_save_path = os.path.join(self.res_img_save_dir,  bg_name+'{}.png'.format(index) )
                mask_save_path = os.path.join(self.res_mask_save_dir, bg_name+'{}.png'.format(index))
                anno_save_path = os.path.join(self.res_anno_save_dir,  bg_name+'{}.txt'.format(index))
                
                if len(pos)>0:
                    utils.write_img(img_gen, bg_save_path)
                    utils.write_img(mask_gen, mask_save_path)
                    with open(anno_save_path, 'w+', encoding='utf-8') as f:
                        for p, label, p_l in zip(pos, ocr_label, p_label):
                            x1,y1,x3,y3 = p 
                            line = str(x1)+','+str(y1)+','+str(x3)+','+str(y1)+','+str(x3)+','+str(y3)+','+str(x1)+','+str(y3)+','+label+','+p_l+'\n'
                            f.write(line)

                print('正在完成{}合成'.format(os.path.basename(bg_path)))
            

    
if __name__ == '__main__':
    config_path = 'sig_gen/config/config.yaml'
    gen_tools = SigGenearation(config_path)
    gen_tools.generation_main()



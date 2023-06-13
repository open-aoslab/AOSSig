import argparse
import logging
import os
import sys
import numpy as np
from skimage import metrics
import torch
import cv2
import torch.nn as nn
from torch import optim
from torch.utils import data
from tqdm import tqdm
import cv2 as cv
from torch.utils.data import DataLoader
from data.SignatureDataset import SignatureDataset
from utils.TensorboardSummary import TensorboardSummary
from utils.loss import ImageBasedCrossEntropyLoss2d, CrossEntropyLoss2d
from torch import optim
import math
from utils.LR_Scheduler import LR_Scheduler
import torch.nn.functional as F 
import json
from utils.metric import MetricCONN,MetricMAD,MetricMSE,MetricGRAD, scores
import networks
import utils
from networks import  SegNet


valid_log = open('log/log.txt', 'a+')
best_pred = 0.0 

def train_net(args,net,device,epochs=5,batch_size=1,lr=0.1,val_percent=0.1,\
              save_cp=True,img_scale=0.5,use_blance_weights= True,\
              n_class=2):

    global best_pred
    kwargs = {
        'num_workers': 0,
        'pin_memory': True
    } 

    save_checkpoints = os.path.join(args.save_root, 'Signature', args.model) 
    utils.utils.make_dir(save_checkpoints)
    save_best_checkpoint = os.path.join(args.save_root, 'Signature', args.model, 'best_pred_checkpoints.tar')

    # build the dataset
    train_dataset = SignatureDataset(dataroot=args.data_root,mode = 'train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    n_train = len(train_dataset)
    print('the numbers of train and val dataset trian_size:%d'%(n_train))
    writer = TensorboardSummary(log_dir=args.model)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = torch.optim.Adam(net.parameters(),lr=lr,)
    lr_sheduler = LR_Scheduler('poly', lr, epochs, n_train)
    criterion  = CrossEntropyLoss2d() 

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i, data in  enumerate(train_loader):
                data['image'] = data['image'].to(device=device)
                data['mask'] = data['mask'].to(device=device)
                
                lr_sheduler(optimizer, i, epoch, best_pred)
                logit_dict = net(data['image'])
                
                losses = criterion(logit_dict, data['mask'].squeeze(1))
                epoch_loss += losses.item()
                writer.add_scalar('Loss/all', losses.item(), global_step)
        
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                pbar.update(data['image'].shape[0]) 
                

                global_step += 1
                if global_step % (len(train_dataset) // (10 * batch_size)) == 0:
                # if True:
                    dataset = 'signature'
                    pred = torch.argmax(F.log_softmax(logit_dict,dim=1),dim=1).unsqueeze(1)
                    writer.visualize_image(dataset,  data['image'],  pred.float() , global_step)
                if global_step%1000==0:
                    print('epoch:{}-global_step:{} loss:{}'.format(epoch, global_step, losses.item()))
   
        torch.save(net.state_dict(), os.path.join(save_checkpoints, 'checkpoint_{}epoch.pth'.format(str(epoch+1))))
        valid_log.flush() 

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    device = torch.device('cuda:0')
    logging.info(f'Using device {device}')
    if args.model == 'deeplabv3+':
        net = networks.modeling.__dict__['deeplabv3plus_resnet50'](num_classes=args.num_classes, output_stride=8)
        networks.convert_to_separable_conv(net.classifier)
        utils.set_bn_momentum(net.backbone, momentum=0.01)
    elif args.model == 'segnet':
        net = SegNet(args.num_classes)
    else:
        pass 

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
   
    try:
        train_net(args, net=net,
                epochs=args.epochs,
                batch_size=args.batchsize,
                lr=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100)
            
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', metavar='E', type=str, default='segnet', help='supporting these models: segnet, deeplabv3+') 
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=20) 
    parser.add_argument('-a', '--save_root', metavar='E', type=str, default='checkpoints') 
    parser.add_argument('-c', '--inchannels', metavar='E', type=int, default=3) 
    parser.add_argument('-n', '--num_classes', metavar='E', type=int, default=2) 
    parser.add_argument('-d', '--data_root', metavar='E', type=str, default='',
                        help='the path of dataset')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0008,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')

    return parser.parse_args()


if __name__ == '__main__':
    main()


    
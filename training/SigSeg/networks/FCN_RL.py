'''
Author: chenshuanghao
Date: 2023-02-24 16:40:40
LastEditTime: 2023-02-24 16:55:23
Description: Do not edit
'''
from torch import nn
from torchvision.models import vgg16
import torch 

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU(inplace=True))
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 宽高减半
    return blk

class VGG16(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG16, self).__init__()
        features = []
        features.extend(vgg_block(2, 3, 64))
        features.extend(vgg_block(2, 64, 128))
        features.extend(vgg_block(3, 128, 256))
        self.index_pool3 = len(features)
        features.extend(vgg_block(3, 256, 512))
        self.index_pool4 = len(features)
        features.extend(vgg_block(3, 512, 512))
        self.features = nn.Sequential(*features)

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)

        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.features.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.features.load_state_dict(new_dict)

    def forward(self, x):
        pool3 = self.features[:self.index_pool3](x)      # 1/8
        pool4 = self.features[self.index_pool3:self.index_pool4](pool3)  # 1/16
        pool5 = self.features[self.index_pool4:](pool4)  # 1/32

        conv6 = self.relu(self.conv6(pool5))  # 1/32
        conv7 = self.relu(self.conv7(conv6))  # 1/32

        return pool3, pool4, conv7

class FCN(nn.Module):
    def __init__(self, num_classes, backbone='vgg'):
        super(FCN, self).__init__()
        if backbone == 'vgg':
            self.features = VGG16()

        self.scores1 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.scores2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.scores3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8)
        self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=4)
        self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        pool3, pool4, conv7 = self.features(x)

        conv7 = self.relu(self.scores1(conv7))  # 1×1卷积将通道数映射为类别数

        pool4 = self.relu(self.scores2(pool4))  # 1×1卷积将通道数映射为类别数

        pool3 = self.relu(self.scores3(pool3))  # 1×1卷积将通道数映射为类别数

        s = pool3 + self.upsample_2x(pool4) + self.upsample_4x(conv7)  # 相加融合
        out_8s = self.upsample_8x(s)  # 8倍上采样

        return out_8s


class FCN_RL(nn.Module):

    def __init__(self, num_classes, mode=None):
        super(FCN_RL, self).__init__()

        self.fcn = FCN(num_classes=num_classes)  
        self.mode = mode

        self.rl = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out_1 = self.fcn(x)
        if self.mode=='train':
            return out_1
        else:
            return self.rl(torch.cat([x,out_1],1)) 

if __name__ == '__main__':
    data = torch.randn([1,3,128, 256])
    net = FCN_RL(num_classes=2, mode='test')
    out = net(data)
    print(out.shape)


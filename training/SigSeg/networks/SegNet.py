import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils import model_zoo
from torchvision import models

class SegNetEnc(nn.Module):

    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

        layers = [
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ]
        layers += [
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        ] * num_layers
        layers += [
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class SegNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        decoders = list(models.vgg16(pretrained=True).features.children())

        self.dec1 = nn.Sequential(*decoders[:5])
        self.dec2 = nn.Sequential(*decoders[5:10])
        self.dec3 = nn.Sequential(*decoders[10:17])
        self.dec4 = nn.Sequential(*decoders[17:24])
      
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.requires_grad = False

        self.enc5 = SegNetEnc(512, 256, 0) # [1, 256, 32,32]
        self.enc4 = SegNetEnc(512, 128, 0) # [1, 128, 64, 64]
        self.enc3 = SegNetEnc(256, 64, 0)  # [1, 64, 128,128]
        self.enc1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, 3, padding=1)

    def forward(self, x):    
        '''
            Attention, input size should be the 32x. 
        '''
        dec1 = self.dec1(x)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
       
        enc5 = self.enc5(dec4) 
        enc4 = self.enc4(torch.cat([dec3, enc5], 1))
        enc3 = self.enc3(torch.cat([dec2, enc4], 1))
        enc1 = self.enc1(torch.cat([dec1, enc3], 1))

        return F.upsample_bilinear(self.final(enc1), x.size()[2:])


if __name__ == '__main__':
    data = torch.Tensor(torch.randn([1,3,256,256]))
    model =  SegNet(3)
    output = model(data)
    print(output.shape)




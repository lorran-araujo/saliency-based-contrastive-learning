""" Models definitions archictures """
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import DWT, IWT, DWTone
import common
import segmentation_models_pytorch as smp
import os.path as osp
import numpy as np
import pandas as pd
from efficientnet_pytorch import EfficientNet
from torchinfo import summary

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class SegNet(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        #self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_1 = nn.Conv2d(self.n_features*8 if self.pool_type == 'gap' else self.n_features*self.multiplier*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)

        # Decoder block 5
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResidualBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self,x):
        x = self.conv_in(x)
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out += residual
        return out


class SegNet_resblock(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_resblock, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        number_of_kernels = 128
        #self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        self.res_block_1 = ResidualBlock(self.n_features, number_of_kernels)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        self.res_block_2 = ResidualBlock(self.n_features*2, number_of_kernels)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        self.res_block_3 = ResidualBlock(self.n_features*4, number_of_kernels)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)
        self.res_block_4 = ResidualBlock(self.n_features*8, number_of_kernels)

        self.conv5_1 = nn.Conv2d(self.n_features*8 if self.pool_type == 'gap' else self.n_features*self.multiplier*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)
        self.res_block_5 = ResidualBlock(self.n_features*8, number_of_kernels)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*8 + self.n_features*2, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*2, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features*2, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(conv5)

        # Decoder block 5
        conv5 = self.res_block_5(conv5)
        x = self.unpool(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4
        conv4 = self.res_block_4(conv4)
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        conv3 = self.res_block_3(conv3)
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        conv2 = self.res_block_2(conv2)
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        conv1 = self.res_block_1(conv1)
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)
        
        
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        #x1 = torch.zeros_like(x1)
        #x2 = torch.zeros(6, 128, 128, 128)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)
        
        
        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x) 
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)
        
        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_two_pools_test(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools_test, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWTone()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*3, self.n_features*2, 1)
        #self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*6, self.n_features*4, 1)
        #self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*12, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*24, self.n_features*8, 1)
        #self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*24, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*24, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*6, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask1_2 = self.pool2(x)

        # Encoder block 2
        x1_h = self.conv2_0_1(x1_h)
        #x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask2_2 = self.pool2(x)
        
        
        # Encoder block 3
        x1_h = self.conv3_0_1(x1_h)
        #x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask3_2 = self.pool2(x)
        
        # Encoder block 4
        x1_h = self.conv4_0_1(x1_h)
        #x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0] 
        x2, mask4_2 = self.pool2(x)

        # Encoder block 5
        x1_h = self.conv5_0_1(x1_h)
        #x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x1_h = x1[1]
        x1 = x1[0]
        x2, mask5_2 = self.pool2(x)
        
        # Decoder block 5
        x1_h = self.conv5_0_1_D(x1_h)
        #x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2, x1_h], dim=1)
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x), dim=1)
        return x
    
    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass



class SegNet_two_pools_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_0_1 = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2 = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_1 = nn.Conv2d(self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*8)



        # Retirar a camada 5 caso fique pior
        self.conv5_0_1_D = nn.Conv2d(self.n_features*32, self.n_features*8, 1)
        self.conv5_0_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, 1)
        self.conv5_3_D = nn.Conv2d(self.n_features*4 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv5_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)
        
        
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_2 = self.pool2(conv2)
        
        
        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(conv4) 
        x2, mask4_2 = self.pool2(conv4)

        # Encoder block 5
        x1 = self.conv5_0_1(x1)
        x2 = self.conv5_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        conv5 = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(conv5)
        x2, mask5_2 = self.pool2(conv5)
        
        # Decoder block 5
        x1 = self.conv5_0_1_D(x1)
        x2 = self.conv5_0_2_D(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.unpool1(x, mask5)
        x = torch.cat([x, conv5], dim=1)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass




class SegNet_two_pools_2(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size

        self.pool1 = DWT()
        self.unpool1 = IWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool2 = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*1 + self.n_features*4, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*2 + self.n_features*8, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*4 + self.n_features*16, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        self.conv5_1 = nn.Conv2d(self.n_features*8 + self.n_features*32, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_2 = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_3 = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(self.n_features*16)

        # Retirar a camada 5 caso fique pior
        self.conv5_3_D = nn.Conv2d(self.n_features*16 + self.n_features*4, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(self.n_features*16)
        self.conv5_2_D = nn.Conv2d(self.n_features*16, self.n_features*16, self.kernel_size, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(self.n_features*16)
        conv5_1_D_out_sz = self.multiplier * 16 * 0.5 if self.pool_type == 'dwt' else 8
        self.conv5_1_D = nn.Conv2d(self.n_features*16, self.n_features*int(conv5_1_D_out_sz), self.kernel_size, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv5_1_D_out_sz))

        
        self.conv4_3_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(x)
        x2, mask1_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(x)
        x2, mask2_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x1, mask3 = self.pool1(x)
        x2, mask3_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x1, mask4 = self.pool1(x)
        x2, mask4_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Encoder block 5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x1, mask5 = self.pool1(x)
        x2, mask5_2 = self.pool2(x)
        x = torch.cat([x1, x2], dim=1)

        # Decoder block 5
        x = self.unpool1(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))       
        
        # Decoder block 4
        x = self.unpool1(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool1(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool1(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool1(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class SegNet_skip(nn.Module):
    # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "dwt", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(SegNet_skip, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)
        
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_3 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(self.n_features*4)
        
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_3 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(self.n_features*8)

        
        self.conv4_3_D = nn.Conv2d(self.n_features*8 + self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2_D = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(self.n_features*8)
        conv4_1_D_out_sz = self.multiplier * 8 * 0.5
        self.conv4_1_D = nn.Conv2d(self.n_features*8, self.n_features*int(conv4_1_D_out_sz), self.kernel_size, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv4_1_D_out_sz))
        
        self.conv3_3_D = nn.Conv2d(self.n_features*4 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2_D = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))
        
        self.conv2_2_D = nn.Conv2d(self.n_features*2 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))
        
        self.conv1_2_D = nn.Conv2d(self.n_features + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()

        

    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        conv3 = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        conv4 = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(conv4)
        
        # Decoder block 4
        x = self.unpool(x, mask4)
        x = torch.cat([x, conv4], dim=1)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3
        x = self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # Decoder block 2
        x = self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        #x = self.conv1_1_D(x)
        x = F.log_softmax(self.conv1_1_D(x))
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


class UNet(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "up", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(UNet, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool = DWT() if self.pool_type == "dwt" else nn.MaxPool2d(2, return_indices=True)
        self.unpool = IWT() if self.pool_type == "dwt" else lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)

        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)

        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)


        conv3_2_D_in_sz = 8 if self.pool_type == 'up' else 2 
        #self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(self.n_features*int(conv3_2_D_in_sz) + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != 'up' else 4
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == 'up' else 2 
        #self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(self.n_features*int(conv2_2_D_in_sz) + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != 'up' else 2
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))

        #self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == 'up' else 1
        self.conv1_2_D = nn.Conv2d(self.n_features*int(conv1_2_D_in_sz) + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()
    


    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(conv1)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(conv2)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x, mask3 = self.pool(conv3)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        
        # Decoder block 3
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        
        # Decoder block 2
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool(x) if self.pool_type == 'up' else self.unpool(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        x = self.conv1_1_D_F(x)
        #x = F.log_softmax(self.conv1_1_D_F(x)) # comentei pq a cross entropy espera sem softmax?
        return x
    

    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            print("Error loading VGG-16 weights in UNet!")
            # Ignore missing keys
            pass


class UNet_two_pools(nn.Module):
    # Unet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    

    def __init__(self, in_channels, out_channels, pretrained=False, pool_type = "up", n_features = 64, kernel_size = 3): # Out channels usually n_classes
        super(UNet_two_pools, self).__init__()
        self.pool_type = pool_type
        self.multiplier = 4 if pool_type == 'dwt' else 1
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.pool1 = DWT()
        self.pool2 = nn.MaxPool2d(2, return_indices=True)
        self.unpool1 = IWT()
        self.unpool2 = nn.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        #self.pool = DWT() if self.pool_type == "dwt" else 
        #self.unpool = IWT() if self.pool_type == "dwt" else nn.interpolate(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(in_channels, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_2 = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(self.n_features)

        self.conv2_0_1 = nn.Conv2d(self.n_features*4, self.n_features*2, 1)
        self.conv2_0_2 = nn.Conv2d(self.n_features, self.n_features*2, 1)
        self.conv2_1 = nn.Conv2d(self.n_features*self.multiplier, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(self.n_features*2)
        self.conv2_2 = nn.Conv2d(self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(self.n_features*2)

        self.conv3_0_1 = nn.Conv2d(self.n_features*8, self.n_features*4, 1)
        self.conv3_0_2 = nn.Conv2d(self.n_features*2, self.n_features*4, 1)
        self.conv3_1 = nn.Conv2d(self.n_features*self.multiplier*2, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(self.n_features*4)
        self.conv3_2 = nn.Conv2d(self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(self.n_features*4)

        self.conv4_0_1 = nn.Conv2d(self.n_features*16, self.n_features*8, 1)
        self.conv4_0_2 = nn.Conv2d(self.n_features*4, self.n_features*8, 1)
        self.conv4_1 = nn.Conv2d(self.n_features*self.multiplier*4, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(self.n_features*8)
        self.conv4_2 = nn.Conv2d(self.n_features*8, self.n_features*8, self.kernel_size, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(self.n_features*8)


        conv3_2_D_in_sz = 8 if self.pool_type == 'up' else 2 
        #self.conv3_2_D = nn.Conv2d(self.n_features*8 + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D = nn.Conv2d(self.n_features*int(conv3_2_D_in_sz) + self.n_features*4, self.n_features*4, self.kernel_size, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(self.n_features*4)
        conv3_1_D_out_sz = self.multiplier * 4 * 0.5 if self.pool_type != 'up' else 4
        self.conv3_1_D = nn.Conv2d(self.n_features*4, self.n_features*int(conv3_1_D_out_sz), self.kernel_size, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv3_1_D_out_sz))

        conv2_2_D_in_sz = 4 if self.pool_type == 'up' else 2 
        #self.conv2_2_D = nn.Conv2d(self.n_features*4 + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D = nn.Conv2d(self.n_features*int(conv2_2_D_in_sz) + self.n_features*2, self.n_features*2, self.kernel_size, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(self.n_features*2)
        conv2_1_D_out_sz = self.multiplier * 2 * 0.5 if self.pool_type != 'up' else 2
        self.conv2_1_D = nn.Conv2d(self.n_features*2, self.n_features*int(conv2_1_D_out_sz), self.kernel_size, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(self.n_features*int(conv2_1_D_out_sz))

        #self.conv1_2_D = nn.Conv2d(self.n_features*2 + self.n_features, self.n_features, self.kernel_size, padding=1)
        conv1_2_D_in_sz = 2 if self.pool_type == 'up' else 1
        self.conv1_2_D = nn.Conv2d(self.n_features*int(conv1_2_D_in_sz) + self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(self.n_features)
        self.conv1_1_D = nn.Conv2d(self.n_features, self.n_features, self.kernel_size, padding=1)
        self.conv1_1_D_bn = nn.BatchNorm2d(self.n_features)
        
        self.conv1_1_D_F = nn.Conv2d(self.n_features, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        if pretrained:
            self.load_weights_VGG16()
    


    def forward(self, x):
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        conv1 = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1, mask1 = self.pool1(conv1)
        x2, mask1_2 = self.pool2(conv1)
           
        # Encoder block 2
        x1 = self.conv2_0_1(x1)
        x2 = self.conv2_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        conv2 = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x1, mask2 = self.pool1(conv2)
        x2, mask2_1 = self.pool2(conv2)


        # Encoder block 3
        x1 = self.conv3_0_1(x1)
        x2 = self.conv3_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        conv3 = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x1, mask3 = self.pool1(conv3)
        x2, mask3_2 = self.pool2(conv3)
        
        # Encoder block 4
        x1 = self.conv4_0_1(x1)
        x2 = self.conv4_0_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        
        # Decoder block 3
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask3)
        x = torch.cat([x, conv3], dim=1)
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        
        # Decoder block 2
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask2)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1
        x = self.unpool2(x) if self.pool_type == 'up' else self.unpool1(x, mask1)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = self.conv1_1_D_bn(F.relu(self.conv1_1_D(x)))
        #x = self.conv1_1_D_F(x)
        x = F.log_softmax(self.conv1_1_D_F(x))
        return x
    



    def load_weights_VGG16(self):
        # Download VGG-16 weights from PyTorch
        vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
        if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
            print("Downloading VGG16 weights")
            weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

        print("Starting mapping VGG16 weights")
        vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
        mapped_weights = {}

        for k_vgg, k_segnet in zip(vgg16_weights.keys(), self.state_dict().keys()):
            if "features" in k_vgg:
                mapped_weights[k_segnet] = vgg16_weights[k_vgg]
                print("Mapping {} to {}".format(k_vgg, k_segnet))
        
        try:
            self.load_state_dict(mapped_weights)
            print("Loaded VGG-16 weights in SegNet !")
        except:
            # Ignore missing keys
            pass


#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================
#==========================================================================================================================================

import torch.nn as nn
from torch.nn import functional as F
import torch

# LANET IMPLEMENTATION

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# ASP module from: 'Multi-scale context aggregation for semantic segmentation of remote sensing images[J]. Remote Sensing.'
class ASP(nn.Module):
    def __init__(self, in_channels, in_stride, reduction=4, RF=(320, 160, 80, 40)):
        super(ASP, self).__init__()
        self.strides = [R // in_stride for R in RF]
        out_channels = in_channels // reduction
        
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, out_channels) for i in range(4)])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+4*out_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels), nn.ReLU()
            )

    def _make_stage(self, in_channels, out_channels):
        #prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size()[2:]
                
        priors = [feats]
        for idx, stage in enumerate(self.stages):
            h_out = h // self.strides[idx]
            w_out = w // self.strides[idx]
            feats_avg = F.adaptive_avg_pool2d(feats, [h_out, w_out])
            feats_avg = stage(feats_avg)
            priors.append(F.interpolate(input=feats_avg, size=(h, w), mode='bilinear', align_corners=True))
        
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

# Patch attention module. Parameters: reduction is the rate of channel reduction. pool_window should be set according to the scaling rate.
class Patch_Attention(nn.Module):
    def __init__(self, in_channels, reduction=8, pool_window=10, add_input=False):
        super(Patch_Attention, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        pool_h = h//self.pool_window
        pool_w = w//self.pool_window
        
        A = F.adaptive_avg_pool2d(x, (pool_h, pool_w))
        A = self.SA(A)
        
        A = F.interpolate(A, (h,w), mode='bilinear')        
        output = x*A
        if self.add_input:
            output += x
        
        return output

# Calculate pixel-wise local attention. Costs more computations.
class Patch_AttentionV2(nn.Module):
    def __init__(self, in_channels, reduction=16, pool_window=10, add_input=False):
        super(Patch_AttentionV2, self).__init__()
        self.pool_window = pool_window
        self.add_input = add_input
        self.SA = nn.Sequential( 
            nn.AvgPool2d(kernel_size=pool_window+1, stride=1, padding = pool_window//2),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        A = self.SA(x)
        
        A = F.interpolate(A, (h,w), mode='bilinear')        
        output = x*A
        if self.add_input:
            output += x
            
        return output

# Attention embedding module. Parameters: reduction is the rate of channel reduction. pool_window should be set according to the scaling rate.
class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, pool_window=6, add_input=False):
        super(Attention_Embedding, self).__init__()
        self.add_input = add_input
        self.SE = nn.Sequential( 
            nn.AvgPool2d(kernel_size=pool_window+1, stride=1, padding = pool_window//2),
            nn.Conv2d(in_channels, in_channels//reduction, 1),
            nn.BatchNorm2d(in_channels//reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels//reduction, out_channels, 1),
            nn.Sigmoid())
            
    def forward(self, high_feat, low_feat):
        b, c, h, w = low_feat.size()
        A = self.SE(high_feat)
        A = F.interpolate(A, (h,w), mode='bilinear')
        
        output = low_feat*A
        if self.add_input:
            output += low_feat
        
        return output

class LANet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(LANet, self).__init__()
        self.FCN = FCN(in_channels, num_classes)
        
        self.PA0 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1),
                                nn.BatchNorm2d(64, momentum=0.95), nn.ReLU(inplace=False),
                                Patch_Attention(64, reduction=8, pool_window=20, add_input=True))
                
        self.PA2 = Patch_Attention(128, reduction=16, pool_window=4, add_input=True)
        self.AE = Attention_Embedding(128, 64)
        
        self.classifier0 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier1 = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x) #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x0 = self.FCN.layer1(x) #size:1/4, C256
        x = self.FCN.layer2(x0) #size:1/8, C512
        x = self.FCN.layer3(x) #size:1/16, C1024
        x = self.FCN.layer4(x) #size:1/16 or 1/32, C2048
        x2 = self.FCN.head(x)  #size:1/16 or 1/32, C128
                
        x2 = self.PA2(x2)
        x0 = self.PA0(x0)
        x0 = self.AE(x2.detach(), x0)
        
        low = self.classifier0(x0)
        low = F.interpolate(low, x_size[2:], mode='bilinear')
        
        high = self.classifier1(x2)
        high = F.interpolate(high, x_size[2:], mode='bilinear')
                
        # high-level and low-level features are auxiliary outputs. Recommended loss: main_loss + 0.3*aux_loss1 + 0.3*aux_loss2
        #return high+low, high
        return high+low, [high, low]

class UNett(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels = 3, classes=8, activation=None):
        super(UNett, self).__init__()
        
        self.base_model = smp.Unet(
                    encoder_name=encoder_name, 
                    encoder_weights=encoder_weights, 
                    in_channels=in_channels, 
                    classes=classes, 
                    activation=activation)
        
        feat_dim = 128

        for class_c in range(classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_' + str(class_c), selector)

        for class_c in range(classes):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__('contrastive_class_selector_memory' + str(class_c), selector)

    def forward(self, x):

        features = self.base_model.encoder(x)  
        decoder_output = self.base_model.decoder(*features) 
        masks = self.base_model.segmentation_head(decoder_output)

        features = features[1:] 
        return masks, features 

class UNett_resized(nn.Module):
    def __init__(self, encoder_name='efficientnet-b0', encoder_weights='imagenet', in_channels = 3, classes=8, activation=None):
        super(UNett_resized, self).__init__()
        
        self.base_model = smp.Unet(
                    encoder_name=encoder_name, 
                    encoder_weights=encoder_weights, 
                    in_channels=in_channels, 
                    classes=classes, 
                    activation=activation)

        self.conv_down = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.conv_up = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        x_downsized = self.conv_down(x) 
        
        features = self.base_model.encoder(x_downsized)  
        decoder_output = self.base_model.decoder(*features) 
        masks = self.base_model.segmentation_head(decoder_output)

        x_upsized = self.conv_up(masks)

        features = features[1:] 
        return x_upsized, features 


def build_model(model_name: str, params: list):
    if model_name == 'unet':
        model = UNett(
            encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['n_classes'],        # model output channels (number of classes in your dataset)
            activation=None,
        )
    elif model_name == 'unet_resized':
        model = UNett_resized(
            encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=params['n_classes'],        # model output channels (number of classes in your dataset)
            activation=None,
        )
    # elif model_name == 'segnet_modificada':
    #     model = SegNet_two_pools_test(in_channels = 3, out_channels = params['n_classes'], pretrained = True, pool_type = 'dwt')
    
    # elif model_name == 'unet2':
    #     model = bigger_UNet(input_channels=3, num_classes=params['n_classes'])

    # elif model_name == 'unet3':
    #     model = UNet(in_channels=3, out_channels=params['n_classes'], pretrained=True, pool_type = 'up', n_features = 128, kernel_size = 3)
    
    # elif model_name == 'unet4':
    #     model = smp.Unet(
    #         encoder_name='efficientnet-b0',      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #         encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #         classes=params['n_classes'],        # model output channels (number of classes in your dataset)
    #         activation=None,
    #     )
    elif model_name =='lanet':
        model = LANet(in_channels=3, num_classes=params['n_classes'], pretrained=True)

    else:
        raise Exception("{} -> invalid model name.".format(model_name))
    
    model.to(params['device'])
    #summary(model, (8,3,256,256)) # mostra arquitetura da rede
    return model

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, H, W] -> [B, 32, H//2, W//2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, H//2, W//2] -> [B, 64, H//4, W//4]
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 64, H//4, W//4] -> [B, 32, H//2, W//2]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, H//2, W//2] -> [B, 3, H, W]
            nn.Sigmoid()  # Para garantir que a imagem reconstruída esteja entre [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math


class CrossAttnMem(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        self.num_heads = num_heads
        self.attention_head_size = embedding_channels 
        self.q_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.q_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.k_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)
        self.v_l2u = nn.Linear(embedding_channels, embedding_channels * self.num_heads, bias=False)

        self.psi = nn.InstanceNorm2d(num_features=1, affine=False)
        self.softmax = Softmax(dim=3)

        self.out_u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.out_l2u = nn.Linear(embedding_channels * self.num_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        self.pseudo_label = None
        self.SMem_size = embedding_channels
        self.register_buffer("queue_ptr", torch.zeros(self.num_class, dtype=torch.long))
        self.register_buffer("kv_queue", torch.randn(self.num_class, self.SMem_size, self.patch_num))

    def multi_head_rep(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):

        emb_s, emb_t = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_t.size()

        q_l2u = self.q_l2u(emb_s)
        k_l2u = self.k_l2u(emb_t)
        v_l2u = self.v_l2u(emb_t)

        batch_size = q_l2u.size(0)

        k_l2u = rearrange(k_l2u, 'b n c -> n (b c)')
        v_l2u = rearrange(v_l2u, 'b n c -> n (b c)')

        k_l2u = repeat(k_l2u, 'n bc -> r n bc', r=batch_size)
        v_l2u = repeat(v_l2u, 'n bc -> r n bc', r=batch_size)

        q_l2u = q_l2u.unsqueeze(1).transpose(-1, -2)
        k_l2u = k_l2u.unsqueeze(1)
        v_l2u = v_l2u.unsqueeze(1).transpose(-1, -2)


        cross_attn_l2u = torch.matmul(q_l2u, k_l2u)
        cross_attn_l2u = self.attn_dropout(self.softmax(self.psi(cross_attn_l2u)))
        cross_attn_l2u = torch.matmul(cross_attn_l2u, v_l2u)

        cross_attn_l2u = cross_attn_l2u.permute(0, 3, 2, 1).contiguous()
        new_shape_l2u = cross_attn_l2u.size()[:-2] + (self.KV_size,)
        cross_attn_l2u = cross_attn_l2u.view(*new_shape_l2u)

        out_l2u = self.out_l2u(cross_attn_l2u)
        out_l2u = self.proj_dropout(out_l2u)

        return out_l2u

class CTSA(nn.Module):
    def __init__(self, num_heads, embedding_channels, channel_num, channel_num_out,
                attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num, embedding_channels, kernel_size=1, padding=0),
                                nn.GELU())
        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = CrossAttnMem(num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num)
        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                    nn.GELU())
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en):
        if not self.training:
            en = torch.cat((en, en))

        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)  #

        emb = self.attn_norm(en)
        emb = self.attn(emb)

        en_s, en_t = torch.split(en, en.size(0) // 2, dim=0)
        emb = emb + en_s

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        if not self.training:
            out = torch.split(out, out.size(0) // 2, dim=0)[0]

        return out

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
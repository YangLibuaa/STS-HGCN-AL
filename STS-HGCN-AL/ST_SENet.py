# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:24:41 2021

@author: liuyu
"""

import torch
import torch.nn as nn
import math
import scipy.io as io
import numpy as np
from torch.autograd import Variable


class SELayer(nn.Module):
    '''
    Original SE block, details refer to "Jie Hu et al.: Squeeze-and-Excitation Networks"
    '''
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias = False),
                                nn.ELU(inplace  = True),
                                nn.Linear(channel // reduction, channel, bias = False),
                                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiScale_Temporal(nn.Module):
    def __init__(self, kernel_size, inc):
        super(MultiScale_Temporal, self).__init__()
        self.conv = nn.Conv2d(in_channels = inc, 
                              out_channels = inc, 
                              kernel_size = (1, kernel_size), 
                              stride = (1, kernel_size), 
                              padding = (0, 0), 
                              bias = False)
        self.bn = nn.BatchNorm2d(inc) 
        self.elu = nn.ELU(inplace = True)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.elu(self.bn(self.conv(x)))
        return output    
    
    
class SENet(nn.Module):
    def __init__(self, inc, outc, kernel_size, reduction = 8):
        super(SENet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels = inc, 
                               out_channels = outc, 
                               kernel_size = (1, kernel_size), 
                               stride = (1, 1), 
                               padding = (0, kernel_size//2), 
                               groups = 2, 
                               bias = False)
        self.bn0 = nn.BatchNorm2d(outc)        
        self.se0 = SELayer(outc, reduction)
        self.conv1 = nn.Conv2d(in_channels = outc, 
                               out_channels = outc, 
                               kernel_size = (1, kernel_size), 
                               stride = (1, 1), 
                               padding = (0, kernel_size//2), 
                               groups = 2, 
                               bias = False)
        self.bn1 = nn.BatchNorm2d(outc)        
        self.se1 = SELayer(outc, reduction)
        self.conv2 = nn.Conv2d(in_channels = outc, 
                               out_channels = outc, 
                               kernel_size = (1, kernel_size), 
                               stride = (1, 1), 
                               padding = (0, kernel_size//2), 
                               groups = 2, 
                               bias = False)
        self.bn2 = nn.BatchNorm2d(outc)        
        self.se2= SELayer(outc, reduction)
        self.elu = nn.ELU(inplace = False)    
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):            
        out = self.elu(self.se0(self.bn0(self.conv0(x))))
        out = self.elu(self.se1(self.bn1(self.conv1(out))))
        out = self.elu(self.se2(self.bn2(self.conv2(out))))
        return out


class Residual_Block(nn.Module): 
    def __init__(self, inc, outc):
        super(Residual_Block, self).__init__()
        
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels = inc, 
                                       out_channels = outc, 
                                       kernel_size = 1, 
                                       stride = 1, 
                                       padding = 0,
                                       bias = False)
        else:
          self.conv_expand = None          
        self.conv1 = nn.Conv2d(in_channels = inc, 
                               out_channels = outc, 
                               kernel_size = (1, 3), 
                               stride = 1, 
                               padding = (0, 1),
                               bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels = outc, 
                               out_channels = outc, 
                               kernel_size = (1, 3), 
                               stride = 1, 
                               padding = (0, 1),
                               bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x): 
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.conv2(output)
        output = self.bn2(torch.add(output,identity_data))
        return output 
    
    
class Input_Layer(nn.Module):
    def __init__(self, inc):
        super(Input_Layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels = 1, 
                                    out_channels = 4, 
                                    kernel_size = (1, 3), 
                                    stride = 1, 
                                    padding = (0, 1), 
                                    bias = False)
        self.bn_input = nn.BatchNorm2d(4)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.bn_input(self.conv_input(x))
        return output


def Embedding_Block(input_block, Residual_Block, num_of_layer, inc, outc):
    layers = []
    layers.append(input_block(inc = inc))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), 
                                     outc = int(math.pow(2, i+1)*outc)))
    return nn.Sequential(*layers) 


class MultiLevel_Spectral(nn.Module): 
    def __init__(self, inc, params_path='./scaling_filter.mat'):
        super(MultiLevel_Spectral, self).__init__()
        self.filter_length = io.loadmat(params_path)['Lo_D'].shape[1]
        self.conv = nn.Conv2d(in_channels = inc, 
                              out_channels = inc*2, 
                              kernel_size = (1, self.filter_length), 
                              stride = (1, 2), padding = 0, 
                              groups = inc, 
                              bias = False)        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis = 1).astype('float32'), np.flip(f['Hi_D'], axis = 1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis = 0)).unsqueeze(1).unsqueeze(1).repeat(inc, 1, 1, 1)            
                m.weight.requires_grad = False 
    
    def self_padding(self, x):
        return torch.cat((x[:, :, :, -(self.filter_length//2-1):], x, x[:, :, :, 0:(self.filter_length//2-1)]), (self.filter_length//2-1))
                           
    def forward(self, x): 
        out = self.conv(self.self_padding(x)) 
        return out[:, 0::2,:, :], out[:, 1::2, :, :]

class ST_SENet(nn.Module):
    def __init__(self, inc, chan_num, si, outc = 64, num_of_layer = 1):
        super(ST_SENet, self).__init__()  
        self.fi = math.floor(math.log2(si))
        self.embedding = Embedding_Block(Input_Layer, 
                                         Residual_Block, 
                                         num_of_layer = num_of_layer, 
                                         inc = inc, 
                                         outc = 4) 
        self.MultiLevel_Spectral = MultiLevel_Spectral(inc = 4*int(math.pow(2, num_of_layer))+inc)
        self.MultiScale_Temporal_gamma = MultiScale_Temporal(pow(2, self.fi-3)//8, 4*int(math.pow(2, num_of_layer))+inc)
        self.MultiScale_Temporal_beta = MultiScale_Temporal(pow(2, self.fi-3)//4, 4*int(math.pow(2, num_of_layer))+inc)
        self.MultiScale_Temporal_alpha = MultiScale_Temporal(pow(2, self.fi-3)//2, 4*int(math.pow(2, num_of_layer))+inc)
        self.MultiScale_Temporal_theta = MultiScale_Temporal(pow(2, self.fi-3), 4*int(math.pow(2, num_of_layer))+inc)
        self.MultiScale_Temporal_delta = MultiScale_Temporal(pow(2, self.fi-3), 4*int(math.pow(2, num_of_layer))+inc)  
        self.gamma_x = SENet(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc = outc//2, kernel_size = 7)
        self.beta_x = SENet(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc = outc//2, kernel_size = 7)
        self.alpha_x = SENet(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc = outc//2, kernel_size = 3)
        self.theta_x = SENet(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc = outc//2, kernel_size = 3)
        self.delta_x = SENet(inc = (4*int(math.pow(2, num_of_layer))+inc)*2, outc = outc//2, kernel_size = 3)        
        self.average_pooling = nn.AdaptiveAvgPool2d((chan_num, 1))
        self.reshapeA = nn.Sequential(nn.Conv2d(in_channels = outc//2, 
                                           out_channels = outc, 
                                           kernel_size = (1, 1), 
                                           stride = (1, 1), 
                                           padding = (0, 0), 
                                           groups = 1, 
                                           bias = False),
                                      nn.BatchNorm2d(outc),
                                      nn.ELU(inplace=False))
        self.reshapeB = nn.Sequential(nn.Conv2d(in_channels = outc//2, 
                                           out_channels = outc, 
                                           kernel_size = (1, 1), 
                                           stride = (1, 1), 
                                           padding = (0, 0), 
                                           groups = 1, 
                                           bias = False),
                                      nn.BatchNorm2d(outc),
                                      nn.ELU(inplace = False))
        self.reshapeD = nn.Sequential(nn.Conv2d(in_channels = outc//2, 
                                           out_channels = outc, 
                                           kernel_size = (1, 1), 
                                           stride = (1, 1), 
                                           padding = (0, 0), 
                                           groups = 1, 
                                           bias = False),
                                      nn.BatchNorm2d(outc),
                                      nn.ELU(inplace = False))
        self.reshapeT = nn.Sequential(nn.Conv2d(in_channels = outc//2, 
                                           out_channels = outc, 
                                           kernel_size = (1, 1), 
                                           stride = (1, 1), 
                                           padding = (0, 0), 
                                           groups = 1, 
                                           bias = False),
                                      nn.BatchNorm2d(outc),
                                      nn.ELU(inplace = False))
        self.reshapeG = nn.Sequential(nn.Conv2d(in_channels = outc//2, 
                                           out_channels = outc, 
                                           kernel_size = (1, 1), 
                                           stride = (1, 1), 
                                           padding = (0, 0), 
                                           groups = 1, 
                                           bias = False),
                                      nn.BatchNorm2d(outc),
                                      nn.ELU(inplace = False))

        
    def forward(self, x):    
        embedding_x = self.embedding(x)      
        cat_x = torch.cat((embedding_x, x), 1)
        for i in range(1, self.fi-2):
            if i <= self.fi-7:
                if i == 1:
                    out, _ = self.MultiLevel_Spectral(cat_x)
                else:
                    out, _ = self.MultiLevel_Spectral(out)
            elif i == self.fi-6:
                if self.fi >= 8:
                    out, gamma = self.MultiLevel_Spectral(out)
                else:
                    out, gamma = self.MultiLevel_Spectral(cat_x)
            elif i == self.fi-5:
                out, beta = self.MultiLevel_Spectral(out)
            elif i == self.fi-4:
                out, alpha = self.MultiLevel_Spectral(out)
            elif i == self.fi-3:
                delta, theta = self.MultiLevel_Spectral(out)
        x1 = torch.cat((self.MultiScale_Temporal_gamma(cat_x), gamma), 1)
        x2 = torch.cat((self.MultiScale_Temporal_beta(cat_x), beta), 1)
        x3 = torch.cat((self.MultiScale_Temporal_alpha(cat_x), alpha), 1)
        x4 = torch.cat((self.MultiScale_Temporal_theta(cat_x), theta), 1)
        x5 = torch.cat((self.MultiScale_Temporal_delta(cat_x), delta), 1)
        x1 = self.gamma_x(x1)
        x2 = self.beta_x(x2)
        x3 = self.alpha_x(x3)
        x4 = self.theta_x(x4)
        x5 = self.delta_x(x5)
        x1 = self.average_pooling(x1)
        x2 = self.average_pooling(x2)
        x3 = self.average_pooling(x3)
        x4 = self.average_pooling(x4)
        x5 = self.average_pooling(x5) 
        x1 = self.reshapeG(x1)
        x2 = self.reshapeB(x2)
        x3 = self.reshapeA(x3)
        x4 = self.reshapeT(x4)
        x5 = self.reshapeD(x5)
        return torch.cat((x1, x2, x3, x4, x5), 1).permute(0, 1, 3, 2).contiguous()


if __name__ == "__main__":
    x = Variable(torch.ones([128, 1, 19, 1280]))
    model = ST_SENet(1, 19, 256)
    output = model(x)
    print(output.size())       
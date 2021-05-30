# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 20:06:11 2020

@author: liuyu
"""
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from ST_SENet import ST_SENet


class rhythmAtt_A(nn.Module):
    def __init__(self, inc, scaling_factor = 16):
        super(rhythmAtt_A, self).__init__()
        self.bn = nn.BatchNorm1d(5)
        self.scaling_factor = scaling_factor
        self.projection1 = nn.Linear(inc, scaling_factor, bias = False)
        self.projection2 = nn.Linear(inc, scaling_factor, bias = False)
        
    def forward(self, x):
        x_norm = self.bn(x.transpose(1, 0)).transpose(1, 0)
        x1, x2 = self.projection1(x_norm), self.projection2(x_norm)
        L = torch.einsum('ij,jk->ik', (x1, x2.transpose(1,0)))/np.sqrt(self.scaling_factor)
        L = torch.softmax(L, -1)
        return L

   
class GATENet(nn.Module):
    def __init__(self, inc, reduction_ratio = 128):
        super(GATENet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(inc, inc // reduction_ratio, bias = False),
                                nn.ELU(inplace = False),
                                nn.Linear(inc // reduction_ratio, inc, bias = False),
                                nn.Tanh(),
                                nn.ReLU(inplace = False))
    
    def forward(self, x):
        y = self.fc(x)
        return y

    
class resGCN(nn.Module):
    def __init__(self, inc, outc):
        super(resGCN, self).__init__()
        self.GConv1 = nn.Conv2d(in_channels = inc, 
                               out_channels = outc, 
                               kernel_size = (1, 1), 
                               stride = (1, 1), 
                               padding = (0, 0), 
                               groups = 5, 
                               bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.GConv2 = nn.Conv2d(in_channels = outc, 
                                out_channels = outc, 
                                kernel_size = (1, 1), 
                                stride = (1, 1), 
                                padding = (0, 0), 
                                groups = 5, 
                                bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.ELU = nn.ELU(inplace = False)
        self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_p, L):
        x = self.bn2(self.GConv2(self.ELU(self.bn1(self.GConv1(x)))))
        y = torch.einsum('bijk,kp->bijp', (x, L))
        y = self.ELU(torch.add(y, x_p))
        return y

    
class classification_net(nn.Module):
    def __init__(self, inc, tmp, outc):
        super(classification_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = inc*5, 
                               out_channels = tmp, 
                               kernel_size = (1, 1), 
                               stride = 1, 
                               padding = (0, 0), 
                               bias = True)
        self.conv2 = nn.Conv2d(in_channels = tmp, 
                               out_channels = outc, 
                               kernel_size = (1, 1), 
                               stride = 1, 
                               padding = (0, 0), 
                               bias = True)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y

class STS_HGCN(nn.Module):
    def __init__(self, dim, chan_num, reduction_ratio, si):
        super(STS_HGCN, self).__init__()
        self.chan_num = chan_num 
        self.dim = dim
        self.ST_SENet = ST_SENet(inc = 1, 
                                 chan_num = chan_num, 
                                 si = si, 
                                 outc = dim, 
                                 num_of_layer = 1)
        self.resGCN1 = resGCN(inc = dim*5, 
                              outc = dim*5)
        self.resGCN2 = resGCN(inc = dim*5, 
                              outc = dim*5)
        self.resGCN3 = resGCN(inc = dim*5, 
                              outc = dim*5)
        self.resGCN4 = resGCN(inc = dim*5, 
                              outc = dim*5)
        self.A = torch.ones((1, self.chan_num*self.chan_num), dtype = torch.float32, requires_grad = False).cuda()
        self.GATENet = GATENet(self.chan_num*self.chan_num, reduction_ratio)

        self.rhythmAtt_G1 = nn.Conv2d(in_channels = dim*5, 
                                      out_channels = dim*5, 
                                      kernel_size = (1, 1), 
                                      stride = 1, 
                                      padding = (0, 0), 
                                      groups = 5, 
                                      bias = False)
        self.bn1 = nn.BatchNorm2d(dim*5)
        self.rhythmAtt_G2 = nn.Conv2d(in_channels = dim*5, 
                                      out_channels = dim*5, 
                                      kernel_size = (1, 1), 
                                      stride = 1, 
                                      padding = (0, 0), 
                                      groups = 5, 
                                      bias = False)
        self.bn2 = nn.BatchNorm2d(dim*5)
        self.rhythmAtt_G3 = nn.Conv2d(in_channels = dim*5, 
                                      out_channels = dim*5, 
                                      kernel_size = (1, 1), 
                                      stride = 1, 
                                      padding = (0, 0), 
                                      groups = 5, 
                                      bias = False)
        self.bn3 = nn.BatchNorm2d(dim*5)
        self.rhythmAtt_G4 = nn.Conv2d(in_channels = dim*5, 
                                      out_channels = dim*5, 
                                      kernel_size = (1, 1), 
                                      stride = 1, 
                                      padding = (0, 0), 
                                      groups = 5, 
                                      bias = False)
        self.bn4 = nn.BatchNorm2d(dim*5)
        self.rhythmAtt_G5 = nn.Conv2d(in_channels = dim*5, 
                                      out_channels = dim*5, 
                                      kernel_size = (1, 1), 
                                      stride = 1, 
                                      padding = (0, 0), 
                                      groups = 5, 
                                      bias = False)
        self.bn5 = nn.BatchNorm2d(dim*5)
        self.rhythmAtt_A = rhythmAtt_A(chan_num)
        self.m1 = nn.Linear(chan_num, 1)
        self.m2 = nn.Linear(chan_num, 1)
        self.m3 = nn.Linear(chan_num, 1)
        self.m4 = nn.Linear(chan_num, 1)
        self.m5 = nn.Linear(chan_num, 1)
        self.classification_net = classification_net(inc = dim, tmp = 32, outc = 2)
        self.ELU = nn.ELU(inplace = False)
        self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for j in m:
                   if isinstance(j, nn.Linear):
                       nn.init.xavier_uniform_(j.weight, gain = 1)
                       
    def forward(self, x):
        x = self.ST_SENet(x)
        s1, s2, s3, s4 = x.size()
        A_ds = self.GATENet(self.A)
        A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        L = torch.einsum('ik,kp->ip', (A_ds, torch.diag(torch.reciprocal(sum(A_ds)))))
        m = self.rhythmAtt_A(torch.cat((self.m1.weight, 
                                        self.m2.weight, 
                                        self.m3.weight, 
                                        self.m4.weight, 
                                        self.m5.weight), 0))
        imp = m.unsqueeze(2).expand_as(torch.zeros(5, 5, self.dim)).contiguous().view(5, -1, 1).unsqueeze(3).expand_as(torch.zeros(5, 5*self.dim, 1, self.chan_num))
        x = self.ELU(self.bn1(sum(torch.einsum('pijk,bijk->bpijk', (imp, x)).split(self.dim, 2)).view(s1, s2, s3, s4)+self.rhythmAtt_G1(x)))
        G1 = self.resGCN1(x, x, L).contiguous()
        G1 = self.ELU(self.bn2(sum(torch.einsum('pijk,bijk->bpijk', (imp, G1)).split(self.dim, 2)).view(s1, s2, s3, s4)+self.rhythmAtt_G2(G1)))
        G2 = self.resGCN2(G1, torch.add(x, G1), L).contiguous()
        G2 = self.ELU(self.bn3(sum(torch.einsum('pijk,bijk->bpijk', (imp, G2)).split(self.dim, 2)).view(s1, s2, s3, s4)+self.rhythmAtt_G3(G2)))
        G3 = self.resGCN3(G2, torch.add(torch.add(x, G1), G2), L).contiguous()
        G3 = self.ELU(self.bn4(sum(torch.einsum('pijk,bijk->bpijk', (imp, G3)).split(self.dim, 2)).view(s1, s2, s3, s4)+self.rhythmAtt_G4(G3)))
        G4 = self.resGCN4(G3, torch.add(torch.add(torch.add(x, G1), G2), G3), L).contiguous()
        G4 = self.ELU(self.bn5(sum(torch.einsum('pijk,bijk->bpijk', (imp, G4)).split(self.dim, 2)).view(s1, s2, s3, s4)+self.rhythmAtt_G5(G4)))
        A, B, C, D, E = G4.split(self.dim, 1)
        y = torch.cat((self.m1(A.view(A.size(0), A.size(1), -1)).unsqueeze(-1).contiguous(),
                       self.m2(B.view(B.size(0), B.size(1), -1)).unsqueeze(-1).contiguous(),
                       self.m3(C.view(C.size(0), C.size(1), -1)).unsqueeze(-1).contiguous(),
                       self.m4(D.view(D.size(0), D.size(1), -1)).unsqueeze(-1).contiguous(),
                       self.m5(E.view(E.size(0), E.size(1), -1)).unsqueeze(-1).contiguous()), 1)

        pred = self.classification_net(y).squeeze()
        return pred, y.squeeze(), L

if __name__ == "__main__":
    x = Variable(torch.ones([128, 1, 18, 1280])).cuda()
    model = STS_HGCN(dim = 64, chan_num = 18, reduction_ratio = 128, si = 256).cuda()
    output = model(x)
    print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))

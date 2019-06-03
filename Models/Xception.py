# -*- coding: utf-8 -*-

"""
@author: Simon Plumecocq
Adapted from Xception by Remi Cadene 
https://github.com/Cadene/pretrained-models.pytorch


Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

The number of Middle flow is a parameter
The Exit flow has been simplified by one conv2d and Average Pooling
Some dropout can be added with rate in parameter

"""
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    """
    def __init__(self, num_middle_block = 8, dropout_rate=[0.3, 0.3, 0.5]):
        """ Constructor
        Args:
            num_middle_block: number of Middle blocks
            dropout_rate: rate for dropout module
                the third value will be used for all middle blocks
        """
        super(Xception, self).__init__()
        self.num_middle_block = num_middle_block
        
        #Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        #self.dropout1 = nn.Dropout2d(dropout_rate[0])
         
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        #self.dropout2 = nn.Dropout2d(dropout_rate[1])
        
        #Middle flow
        rep = []
        for i in range(num_middle_block):
            rep.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
            #rep.append(nn.Dropout2d(dropout_rate[2]))
        self.rep = nn.Sequential(*rep)
        
        #Updated Exit flow
        self.conv3 = nn.Conv2d(728, 10, kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(10) 
        self.relu3 = nn.ReLU(inplace=True)
        
        self.avPool = nn.AdaptiveAvgPool2d(output_size=(1,1))        
    

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        #x = self.dropout1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        #x = self.dropout2(x)
        
        x = self.rep(x)
                
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.avPool(x)
      
        return x.view(x.size()[0],x.size()[1])



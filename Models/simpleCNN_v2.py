# -*- coding: utf-8 -*-
"""
Created on May 2019

@author: Simon Plumecocq
"""
import torch
import torch.nn as nn

class StandardConv(torch.nn.Module):
    ''' Conv2d followed by BatchNorm2d and Relu '''
    def __init__(self,input_frames,output_frames,kernel_size,stride,padding):
        super(StandardConv, self).__init__()
        self.conv = nn.Conv2d(input_frames,output_frames,kernel_size,stride,padding)        
        self.bn = nn.BatchNorm2d(output_frames)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        x = self.relu(x) 
        return x 
       
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = StandardConv(3, 42, kernel_size=5,stride=2,padding=2)  
        self.conv2 = StandardConv(42, 42, kernel_size=3,stride=1,padding=1)
        self.maxPool2 = nn.MaxPool2d(2, 2)

        self.conv3 = StandardConv(42, 84, kernel_size=3,stride=1,padding=1)
        self.maxPool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = StandardConv(84, 10, kernel_size=1,stride=1,padding=0)
        
        self.avPool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        
        x = self.conv3(x)
        x = self.maxPool3(x)
        
        x = self.conv4(x)
        
        x = self.avPool(x)
        
        return x.view(x.size()[0],x.size()[1])
    
    
        
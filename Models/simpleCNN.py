# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:30:41 2019

@author: Plumecocq
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 42, kernel_size=5,stride=2,padding=2)
        self.bn1 = nn.BatchNorm2d(42)     
        self.conv2 = nn.Conv2d(42, 42, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(42) 
        self.maxPool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(42, 84, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(84) 
        self.maxPool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(84, 10, kernel_size=1,stride=1,padding=0)
        self.bn4 = nn.BatchNorm2d(10) 
        
        self.avPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        #self.softMax = torch.nn.Softmax()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxPool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxPool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.avPool(x)
        
        return x.view(x.size()[0],x.size()[1])
    
    
        
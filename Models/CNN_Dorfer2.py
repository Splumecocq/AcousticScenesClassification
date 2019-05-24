# -*- coding: utf-8 -*-
"""
Created on May 2019

@author: Plumecocq

CNN of Dorfer team presented to the DCASE 2018

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

class CNN_Dorfer2(torch.nn.Module):
    def __init__(self):
        super(CNN_Dorfer2, self).__init__()
        self.conv1 = nn.Conv2d(3, 42, kernel_size=5,stride=2,padding=2)
        self.bn1 = nn.BatchNorm2d(42)     
        self.conv2 = nn.Conv2d(42, 42, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(42) 
        self.maxPool2 = nn.MaxPool2d(2, 2)
        self.gn2 = tdist.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self.conv3 = nn.Conv2d(42, 84, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(84)
        self.conv4 = nn.Conv2d(84, 84, kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(84) 
        self.maxPool4 = nn.MaxPool2d(2, 2)
        self.gn4 = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.75]))
        
        self.conv5 = nn.Conv2d(84, 168, kernel_size=3,stride=1,padding=1)
        self.bn5 = nn.BatchNorm2d(168)
        self.dropout5 = nn.Dropout2d(0.3)
        self.conv6 = nn.Conv2d(168, 168, kernel_size=3,stride=1,padding=1)
        self.bn6 = nn.BatchNorm2d(168)
        self.dropout6 = nn.Dropout(0.3)
        self.conv7 = nn.Conv2d(168, 168, kernel_size=3,stride=1,padding=1)
        self.bn7 = nn.BatchNorm2d(168)
        self.dropout7 = nn.Dropout(0.3)
        self.conv8 = nn.Conv2d(168, 168, kernel_size=3,stride=1,padding=1)
        self.bn8 = nn.BatchNorm2d(168)
        self.maxPool8 = nn.MaxPool2d(2, 2)
        self.gn8 = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.75]))
        
        self.conv9 = nn.Conv2d(168, 336, kernel_size=3,stride=1,padding=0)
        self.bn9 = nn.BatchNorm2d(336)        
        self.dropout9 = nn.Dropout(0.5)
        self.conv10 = nn.Conv2d(336, 336, kernel_size=1,stride=1,padding=0)
        self.bn10 = nn.BatchNorm2d(336)        
        self.dropout10 = nn.Dropout(0.5)
        
        self.conv11 = nn.Conv2d(336, 10, kernel_size=1,stride=1,padding=0)
        self.bn11 = nn.BatchNorm2d(10) 
        self.gn11 = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.3]))
        
        self.avPool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxPool2(x)
        if self.training:
            x = x + self.gn2(x.size())
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxPool4(x)    
        if self.training:
            x = x + self.gn4(x.size())
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout6(x)
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout7(x)
        x = F.relu(self.bn8(self.conv8(x)))   
        x = self.maxPool8(x)
        if self.training:
            x = x + self.gn8(x.size())
        
        x = F.elu(self.bn9(self.conv9(x)))
        x = self.dropout9(x)
        x = F.elu(self.bn10(self.conv10(x)))
        x = self.dropout10(x)
        
        x = F.relu(self.bn11(self.conv11(x)))
        if self.training:
            x = x + self.gn11(x.size())
        x = self.avPool(x)
        
        return x.view(x.size()[0],x.size()[1])
        

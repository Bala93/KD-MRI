import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math


#### Super-resolution vdsr 

class TeacherVDSR(nn.Module):
    
    def __init__(self):

        super(TeacherVDSR, self).__init__()
        
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv8 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv9 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv10 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv11 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x7 = F.relu(self.conv7(x6))
        x8 = F.relu(self.conv8(x7))
        x9 = F.relu(self.conv9(x8))
        x10 = F.relu(self.conv10(x9))
        x11 = self.conv11(x10)

        x11 = x + x11 

        return x6,x11


class StudentVDSR(nn.Module):
    
    def __init__(self):

        super(StudentVDSR, self).__init__()
        
        self.conv1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.conv7 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
       
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x7 = self.conv7(x6)

        x7 = x7 + x 

        return x4,x7


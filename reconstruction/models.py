import torch
from torch import nn
from torch.nn import functional as F
import os 
import numpy as np
import math

class DataConsistencyLayer(nn.Module):

    def __init__(self,us_mask):
        
        super(DataConsistencyLayer,self).__init__()

        self.us_mask = us_mask 

    def forward(self,predicted_img,us_kspace):

        # us_kspace     = us_kspace[:,0,:,:]
        predicted_img = predicted_img[:,0,:,:]
        
        kspace_predicted_img = torch.rfft(predicted_img,2,True,False).double()
        #print (us_kspace.shape,predicted_img.shape,kspace_predicted_img.shape,self.us_mask.shape)
        #torch.Size([4, 1, 256, 256, 2]) torch.Size([4, 256, 256]) torch.Size([4, 256, 256, 2]) torch.Size([1, 256, 256, 1])
        #print (self.us_mask.dtype,us_kspace.dtype)
        updated_kspace1  = self.us_mask * us_kspace 
        updated_kspace2  = (1 - self.us_mask) * kspace_predicted_img
        #print("updated_kspace1 shape: ",updated_kspace1.shape," updated_kspace2 shape: ",updated_kspace2.shape)
        #updated_kspace1 shape:  torch.Size([4, 1, 256, 256, 2])  updated_kspace2 shape:  torch.Size([4, 256, 256, 2])
        updated_kspace   = updated_kspace1[:,0,:,:,:] + updated_kspace2

        updated_img    = torch.ifft(updated_kspace,2,True) 
        
        #update_img_abs = torch.sqrt(updated_img[:,:,:,0]**2 + updated_img[:,:,:,1]**2)
        update_img_abs = updated_img[:,:,:,0]
        
        update_img_abs = update_img_abs.unsqueeze(1)
        
        return update_img_abs.float()


class TeacherNet(nn.Module):
    
    def __init__(self):
        super(TeacherNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = self.conv5(x4)
        
        return x1,x2,x3,x4,x5


class DCTeacherNet(nn.Module):

    def __init__(self,args):

        super(DCTeacherNet,self).__init__()

        self.cascade1 = TeacherNet()
        self.cascade2 = TeacherNet()
        self.cascade3 = TeacherNet()
        self.cascade4 = TeacherNet()
        self.cascade5 = TeacherNet()

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device).double()
        self.dc = DataConsistencyLayer(us_mask)

    def forward(self,x,x_k):

        x1 = self.cascade1(x) # list of channel outputs 
        x1_dc = self.dc(x1[-1],x_k)

        x2 = self.cascade2(x1_dc)
        x2_dc = self.dc(x2[-1],x_k)

        x3 = self.cascade3(x2_dc)
        x3_dc = self.dc(x3[-1],x_k)

        x4 = self.cascade4(x3_dc)
        x4_dc = self.dc(x4[-1],x_k)

        x5 = self.cascade5(x4_dc)
        x5_dc = self.dc(x5[-1],x_k)

        return x1,x2,x3,x4,x5,x5_dc


class StudentNet(nn.Module):
    
    def __init__(self):
        super(StudentNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        
        return x1,x2,x3


class DCStudentNet(nn.Module):

    def __init__(self,args):

        super(DCStudentNet,self).__init__()

        self.cascade1 = StudentNet()
        self.cascade2 = StudentNet()
        self.cascade3 = StudentNet()
        self.cascade4 = StudentNet()
        self.cascade5 = StudentNet()

        us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)
        us_mask = us_mask.double()

        self.dc = DataConsistencyLayer(us_mask)

    def forward(self,x,x_k):

        x1 = self.cascade1(x) # list of channel outputs 
        x1_dc = self.dc(x1[-1],x_k)

        x2 = self.cascade2(x1_dc)
        x2_dc = self.dc(x2[-1],x_k)

        x3 = self.cascade3(x2_dc)
        x3_dc = self.dc(x3[-1],x_k)

        x4 = self.cascade4(x3_dc)
        x4_dc = self.dc(x4[-1],x_k)

        x5 = self.cascade5(x4_dc)
        x5_dc = self.dc(x5[-1],x_k)

        return x1,x2,x3,x4,x5,x5_dc



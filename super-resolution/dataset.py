import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from utils import npComplexToTorch

class SR_SliceData(Dataset):

    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self,root):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        self.mask = np.zeros([256,256])
        self.mask[128-32:128+32,128-32:128+32] = 1

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            target = data['volfs'][:,:,slice]
            kspace = np.fft.fft2(target,norm='ortho')
            kspace_shifted = np.fft.fftshift(kspace)
            truncated_kspace = self.mask * kspace_shifted

            lr_img = np.abs(np.fft.ifft2(truncated_kspace,norm='ortho'))

            return torch.from_numpy(lr_img),torch.from_numpy(target)

class SR_SliceDataDev(Dataset):

    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self,root):

        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []

        self.mask = np.zeros([256,256])
        self.mask[128-32:128+32,128-32:128+32] = 1

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice = self.examples[i] 
    
        with h5py.File(fname, 'r') as data:

            target = data['volfs'][:,:,slice]
            kspace = np.fft.fft2(target,norm='ortho')
            kspace_shifted = np.fft.fftshift(kspace)
            truncated_kspace = self.mask * kspace_shifted

            lr_img = np.abs(np.fft.ifft2(truncated_kspace,norm='ortho'))

            return torch.from_numpy(lr_img),torch.from_numpy(target),str(fname.name),slice


 
 

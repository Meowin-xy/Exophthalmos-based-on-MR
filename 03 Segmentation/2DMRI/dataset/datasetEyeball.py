"""
@author: zhengyong Huang
Time: 2024-12-02 
"""
import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from natsort import natsorted
from scipy.ndimage.interpolation import zoom


class datasetEyeball(Dataset):
    def __init__(self, data_root, transform=None, mode='train'):
        self.sample_list_A = [s for s in data_root]
        self.mode = mode
        self.transform = transform  # using transform in torch!
        

    def __len__(self):
        return len(self.sample_list_A)

    def __getitem__(self, idx):
        sample = dict()
        data_path = self.sample_list_A[idx]
        label_path = data_path.replace('images', 'masks')

        # print('data_path: ', data_path)
        # print('label_path: ', label_path)
        
        sample['id'] = data_path.split('/')[-1][:-7]  # brats

        np_data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        np_label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        if np_data.shape[1]<111:
            print('id: ', sample['id'])
            print("data shape: ", np_data.shape)
            print("label shape: ", np_label.shape)

        if np_data.shape[1]<111:
            np_data = np.pad(np_data, ((0, 0), (0, 113-np_data.shape[1]), (0, 0)), mode='constant', constant_values=0)
            np_label = np.pad(np_label, ((0, 0), (0, 113-np_label.shape[1]), (0, 0)), mode='constant', constant_values=0)

        np_data = np_data[:, :112, :]
        np_label = np_label[:, :112, :]

        np_data = np.expand_dims(np_data, -1)
        np_label = np.expand_dims(np_label, -1)

        sample['input'] = np_data
        sample['target'] = np_label

        if self.transform:
            sample = self.transform(sample)

        return sample

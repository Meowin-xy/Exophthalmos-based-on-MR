import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class datasetEyeball(Dataset):
    def __init__(self, data_root, transform=None, mode='test'):
        self.sample_list_A = [s for s in data_root]
        self.mode = mode
        self.transform = transform  # using transform in torch!

    def __len__(self):
        return len(self.sample_list_A)

    def __getitem__(self, idx):
        sample = dict()
        data_path = self.sample_list_A[idx]

        sample['id'] = data_path.split('/')[-1][:-7]  # brats

        np_data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        if np_data.shape[1]<120:
            print('id: ', sample['id'])
            print("data shape: ", np_data.shape)

        if np_data.shape[1]<112:
            np_data = np.pad(np_data, ((0, 0), (0, 113-np_data.shape[1]), (0, 0)), mode='constant', constant_values=0)

        np_data = np_data[:, :112, :]
        np_data = np.expand_dims(np_data, -1)
        
        sample['input'] = np_data

        if self.transform:
            sample = self.transform(sample)
            
        return sample
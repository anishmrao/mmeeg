
from torch.utils.data import Dataset
import os
import numpy as np
import scipy.io
import torch
import torch
from .registry import DATASETS

@DATASETS.register_module()
class EEGDataset(Dataset):
    def __init__(self, data_root, subs, batch_size, augment=True):
        self.root = data_root
        self.subs = subs
        self.batch_size = batch_size
        self.augment = augment
        self.allData=None
        self.allLabel=None
        self.create_dataset()
    
    def get_source_data(self, sub):
        total_data = scipy.io.loadmat(os.path.join(self.root, '%d.mat' % sub))
        train_data = total_data['data']
        train_label = total_data['label']

        train_data = np.transpose(train_data, (2, 1, 0))
        train_data = np.expand_dims(train_data, axis=1)
        train_label = np.transpose(train_label)

        allData = train_data
        allLabel = train_label[0]

        shuffle_num = np.random.permutation(len(allData))
        allData = allData[shuffle_num, :, :, :]
        allLabel = allLabel[shuffle_num]

        # standardize
        target_mean = np.mean(allData)
        target_std = np.std(allData)
        allData = (allData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return allData, allLabel

    def create_dataset(self):
        for sub in self.subs:
            allData, allLabel = self.get_source_data(sub)
            if(self.allData is None):
                self.allData = allData
                self.allLabel = allLabel
            else:
                self.allData = np.concatenate((self.allData, allData), axis=0)
                self.allLabel = np.concatenate((self.allLabel, allLabel), axis=0)
        
        self.allData = torch.from_numpy(self.allData)
        self.allLabel = torch.from_numpy(self.allLabel)
        
        if(self.augment):
            aug_data, aug_label = self.interaug(self.allData, self.allLabel)
            self.allData = torch.cat((self.allData, aug_data))
            self.allLabel = torch.cat((self.allLabel, aug_label))

    def __len__(self):
        return len(self.allData)
    
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 64, 480))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 60:(rj + 1) * 60] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 60:(rj + 1) * 60]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data)
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label)
        aug_label = aug_label.long()
        return aug_data, aug_label

    def __getitem__(self, idx):
        return self.allData[idx].type(torch.FloatTensor), self.allLabel[idx]
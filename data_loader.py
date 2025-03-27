'''
Author: syu-kylin 79462188+syu-kylin@users.noreply.github.com
Date: 2024-06-24 13:49:59
LastEditors: syu-kylin
LastEditTime: 2024-08-18 20:17:26
FilePath: \4d-HIE-HRV Delphi\data_loader.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import numpy as np
import pandas as pd
import random
import scipy.io as scio    # read .mat file
from scipy import signal
import logging

import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.data import SubsetRandomSampler

from utils import Params
from project_init import project_init

logger = logging.getLogger('project_log')

def read_split_data(window_length, seed_epoch):
    """ Read the rr struct file and split them as train and validation set.
    Args:
        #window_length(int or str): the window length of the rr in mins (2 or 5).
        #seed_epoch(int): a random seed to make the split train and val epoch id reproduce.
    Return:
        train_epochs(np struct): split train epochs.
        val_epochs(np struct): split train epochs.
    """

    # 1). read the rr struct file
    # home_dir = '/home/shuwenyu'
    home_dir = '/mnt/files'
    data_dir = '{}/Datasets/Delphi/RPeaks/{}mins'.format(home_dir, window_length)
    train_dataset_name = ['ANSeR1_weak_7-11h', 'ANSeR1_weak_13-23h', 'ANSeR1_weak_25-35h', 'ANSeR1_weak_37-47h',  
                          'ANSeR2_weak_7-11h', 'ANSeR2_weak_13-23h', 'ANSeR2_weak_25-35h', 'ANSeR2_weak_37-47h', 
                          'ANSeR2_strong_6-48h']
    # train_dataset_name = ['ANSeR2_weak_7-11h', 'ANSeR2_weak_13-23h', 'ANSeR2_weak_25-35h', 'ANSeR2_weak_37-47h', 
    #                       'ANSeR2_strong_6-48h']
    test_dataset_name = ['ANSeR1_strong_6-48h']
    NN_epochs_dev_lst, NN_epochs_test_lst = [], []

    dataset_name_info = 'train dataset: '
    for dataname in train_dataset_name:
        fn = '{}/NN_epoch_{}_{}min_std_0.12.mat'.format(data_dir, dataname, window_length)
        epoch_dict = scio.loadmat(fn)
        NN_epochs = epoch_dict['NN_epoch_ANSeR'][0]
        n_files = len(NN_epochs['file_id'])
        NN_epochs_dev_lst.append(NN_epochs)

        dataset_name_info += f'{dataname}: {n_files}h, '

    dataset_name_info = dataset_name_info[:-2] + '.\n'
    dataset_name_info += 'test dataset: '
    for dataname in test_dataset_name:
        fn = '{}/NN_epoch_{}_{}min_std_0.12.mat'.format(data_dir, dataname, window_length)
        epoch_dict = scio.loadmat(fn)
        NN_epochs = epoch_dict['NN_epoch_ANSeR'][0]
        n_files = len(NN_epochs['file_id'])
        NN_epochs_test_lst.append(NN_epochs)

        dataset_name_info += f'{dataname}: {n_files}h, '

    dataset_name_info = dataset_name_info[:-2] + '.\n'

    NN_epochs_dev = np.concatenate(NN_epochs_dev_lst)
    NN_epochs_test = np.concatenate(NN_epochs_test_lst)
    N_files_dev = len(NN_epochs_dev['file_id'])
    N_epochs_test = len(NN_epochs_test['file_id'])
    logger.info(dataset_name_info)

    # 2). remove these invalid dimensions (that produced from the saved files)
    dt = np.dtype([('file_id', '<U20'), ('rr_epochs', 'O'), ('EEG_grade', 'i4'), ('n_epochs', 'i4')])
    NN_epochs_dev_org = np.zeros((N_files_dev), dtype=dt)
    NN_epochs_test_org = np.zeros((N_epochs_test), dtype=dt)

    for i in range(N_files_dev):
        file_id = NN_epochs_dev[i]['file_id'][0]
        grade = NN_epochs_dev[i]['EEG_grade'][0][0]
        n_epoch = NN_epochs_dev[i]['n_epochs'][0][0]
        rr_epoch = NN_epochs_dev[i]['rr_epochs']

        NN_epochs_dev_org[i] = (file_id, rr_epoch, grade, n_epoch)

    for i in range(N_epochs_test):
        file_id = NN_epochs_test[i]['file_id'][0]
        grade = NN_epochs_test[i]['EEG_grade'][0][0]
        n_epoch = NN_epochs_test[i]['n_epochs'][0][0]
        rr_epoch = NN_epochs_test[i]['rr_epochs']

        NN_epochs_test_org[i] = (file_id, rr_epoch, grade, n_epoch)
        
    # 3). split train and validation set
    random.seed(seed_epoch)
    valid_epochs_idx = random.sample(range(N_files_dev), 500)
    # valid_epochs_idx = random.sample(range(N_files_dev), int(N_files_dev*0.2))
    # print(f'validation set epoch ids:\n {valid_epochs_idx}')

    val_epochs = NN_epochs_dev_org[valid_epochs_idx]
    # train_epochs = NN_epochs_dev_org
    train_epochs = np.delete(NN_epochs_dev_org, valid_epochs_idx)
    test_epochs = NN_epochs_test_org

    message = ''.join([
        '-'*45, '\n', 'Data preprocessing\n\n',
        f'Number of 1h epoch files on development set: {N_files_dev}\n',
        f'Number of 1h epoch files on test set: {len(NN_epochs_test)}\n',
        f'validation set epoch indexs:\n{valid_epochs_idx}\n\n',
        f'Number of 1h epoch from train set: {len(train_epochs)}\n',
        f'Number of 1h epoch from validation set {len(val_epochs)}\n',
        f'Number of 1h epoch from test set: {len(test_epochs)}\n\n',
    ])
    logger.info(message)

    return train_epochs, val_epochs, test_epochs

class NormalizeAndToTensor:
    def __init__(self, mean: float=0.0, std: float=0.0,
                 min: float=0.0, max: float=0.0, min_max_enable: bool=False):
        """
        Args:
            mean (float): Mean value for normalization.
            std (float): Standard deviation for normalization.
            min (float): Min value for min-max normalization.
            max (float): Max value for min-max normalization.
            min_max (bool): Whether to use min-max normalization.
        """
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        
        if min_max_enable:
            self.normalize = self.minmax_norm
            logger.info('\033[35;1mdata transformed with min-max normalization.\033[0m')
        else:
            self.normalize = self.stand_norm
            logger.info('\033[35;1mdata transformed with 0 mean and 1 std.\033[0m')

    def minmax_norm(self, sample):
        sample = (sample - self.min) / (self.max - self.min)
        return sample
    def stand_norm(self, sample):
        sample = (sample - self.mean) / self.std
        return sample

    def __call__(self, sample, label):
        """
        Args:
            sample (np.ndarray): Input signal sample.
            label (int): Label of the signal sample.
        Returns:
            torch.Tensor: Normalized signal sample as a tensor.
        """
        sample = self.normalize(sample)
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label).long()
        return sample, label
    


class SignalDataset(Dataset):

    def __init__(self, epoch_struct, set_name, transform=None):
        self.data, self.labels = self.build_sampleset(epoch_struct, set_name)
        self.transform = transform

    def build_sampleset(self, epoch_struct, set_name):
        """
        This function is use to concatenate the all rr one-hour epochs. Weak labels are allocated as their global labels.

        Args:
            #epoch_struct (numpy struct): numpy struct of rr epochs, each of them 
            consists of approximatly one hour rr with 5 or 2 mins windows.
            #set_name (str): 'pretrain' or 'finetune', for the finetune dataset, the classes weights will be reported.
        """
        n_files = len(epoch_struct)
        
        Grades = []
        n_epoch_window = 0
        for i in range(n_files):  
            num_epoch_1h = epoch_struct[i]['n_epochs']
            n_epoch_window += num_epoch_1h
            
            grade = epoch_struct[i]['EEG_grade']
            Grades.append(grade)
        
        window_length = len(epoch_struct[0]['rr_epochs'][0])
        rr_data = np.zeros((n_epoch_window, window_length))
        HIE_grades = np.zeros(n_epoch_window)
        file_ids = np.zeros((n_epoch_window), dtype='<U20')
        start_idx = 0 
        for i in range(n_files):
            epoch_one_file = epoch_struct[i]['rr_epochs']
            grade_one_file = Grades[i]
            n_epoch = epoch_struct[i]['n_epochs']
            file_id = epoch_struct[i]['file_id']
            grade = epoch_struct[i]['EEG_grade']
            
            end_idx = start_idx + n_epoch
            
            rr_data[start_idx:end_idx] = epoch_one_file
            HIE_grades[start_idx:end_idx] = grade_one_file
            file_ids[start_idx:end_idx] = file_id
            start_idx = end_idx
        
        labels = HIE_grades
        labels = labels.astype(int)
        assert set(labels) == {0, 1}, f'Invalid labels: {set(labels)}'        

        pos_perct = sum(labels == 1)/len(labels)
        neg_perct = sum(labels == 0)/len(labels)
        message = ''.join([
            f'Number of samples in {set_name} set: {n_epoch_window}.\n',
            f'Shape of rr data: {rr_data.shape}.\n\n',
            f'{set_name} set class distribution:\n',
            f'Percent of class 0: {neg_perct:.4f}\n',
            f'Percent of class 1: {pos_perct:.4f}\n',
        ])
        logger.info(message)

        return rr_data, labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        sample = np.expand_dims(sample, axis=0)
        # print(f'Sample shape before transform: {sample.shape}, dtype: {sample.dtype}')
        if self.transform:
            sample, label = self.transform(sample, label)
        # print(f'Sample shape after transform: {sample.shape}, dtype: {sample.dtype}')
        return sample, label




if __name__ == "__main__":

    job_name = 'Test'
    group_name = 'test'
    # config_json_path = '../log/4d-HRV ANSeR//HrvConfermer/{}/{}//project_init_config.json'.format(job_name, group_name)

    config_json_path = project_init(job_name, group_name)
    config = Params(config_json_path)

    logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s - %(message)s',
                    # filemode='w',
                    # filename=f'RPeaks_stats.log',
                    # filename=f'RPeaks_stats_whole.log'
                    )

    window_length, seed_epoch = config.window_length, config.seed_epoch
    train_epochs, val_epochs, test_epochs = read_split_data(window_length, seed_epoch)
    signal_transform = NormalizeAndToTensor(mean=config.mean, std=config.std, 
                                                    min=config.min, max=config.max, 
                                                    min_max_enable=config.min_max_enable)    
    train_dataset = SignalDataset(train_epochs, 'train', signal_transform)
    val_dataset = SignalDataset(val_epochs, 'validation', signal_transform)
    test_dataset = SignalDataset(test_epochs, 'test', signal_transform)

    
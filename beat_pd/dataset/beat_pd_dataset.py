from torch.utils.data import Dataset
import pandas as pd
import os

import numpy as np
import torch

from .utils import convert_raw_to_timeseries

class BeatPD_Dataset(Dataset):
    def __init__(self, data_folder: str, 
                 label_data: pd.DataFrame,
                 transforms=None,
                 target_transforms=None,
                 interpolate=False
                 ):
        ''' Dataset class for Beat-PD challenge
        
        data_folder: Folder store csv of timeseries data
        label_data: Dataframe contains label.
        
        transforms: a function that take a timeseries data (torch Tensor) and transforms it to another timeseries data
        target_transforms: a function that take target (torch Tensor) and transforms it

        # Get Item: Return X and y
        '''
        self.data_folder = data_folder
        self.label_data = label_data
        
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.interpolate = interpolate
        ## 
        self.key_list = ['on_off', 'dyskinesia', 'tremor']

    def get_sample(self, i: int):
        sample_label_data = self.label_data.iloc[i]
        
        raw_file_name = sample_label_data.measurement_id + '.csv'
        raw_data_path = os.path.join(self.data_folder, raw_file_name)
        raw_data = pd.read_csv(raw_data_path)

        raw_timeseries = convert_raw_to_timeseries(raw_data, self.interpolate)
        targets = torch.tensor([raw_data[k] for k in self.key_list])
        
        return raw_timeseries, targets

    def __getitem__(self, i):
        raw_timeseries, targets = self.get_sample(i)

        if self.transforms:
            raw_timeseries = self.transforms(raw_timeseries)
        
        if self.target_transforms:
            targets = self.target_transforms(targets)
        
        return raw_timeseries, targets

    def __len__(self):
        return self.label_data.shape[0]
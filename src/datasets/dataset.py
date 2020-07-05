import torch
import os
from torch.utils.data import Dataset
import torch.nn.functional as F

import config

'''
    Wrapper around torch.utils.data.Dataset
    - handles specific needs for our data (like padding, path organization, etc.)
'''
class Dataset(Dataset):
    def __init__(self, dataset_id=config.DATASET_EMERGENCY,
                 sample_rate=48000, 
                 split_mode="training", 
                 fixed_padding=True):
        
        assert split_mode in ["training", "validation", "testing"]
        self.sample_rate = sample_rate
        self.fixed_padding = fixed_padding
        
        if sample_rate == 8000:
            self.max_length_sample = 80249 # manually determined
        elif sample_rate == 48000:
            self.max_length_sample = 481489 # manually determined
        
        dataset_directory = config.DATASETS_DIR[dataset_id][sample_rate]
        print(f"Loading cached {split_mode} data of dataset {dataset_id} from {dataset_directory}")

        self.dataset = torch.load(os.path.join(dataset_directory, f"{split_mode}.pt"))
    
    def __getitem__(self, index):
        audio = self.dataset[index]['data'][0]
        if self.fixed_padding:
            assert self.max_length_sample - len(audio) > 0
            audio = F.pad(audio, (0, self.max_length_sample - len(audio)), mode='constant', value=0)

        return {'audio': audio, 
                'sample_rate': self.sample_rate, 
                'label': 1 if self.dataset[index]['binary_class'] == 'positive' else 0}
    
    def __len__(self):
        return len(self.dataset)
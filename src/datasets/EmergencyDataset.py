import os
import torch
from torch.utils.data import Dataset
import config

'''
To create the .pt-files from scratch (i.e., run whole pre-processing pipeline: cropping/padding, normalizing, downsampling, etc.): run the file `lib dat dat`a PrepareData.py` in branch `pascal_lib`
'''

def load_preprocessed_data_from_cache(mode = "train"):
    
    directory = config.DATA_8K_DIR
    print("Loading cached {} data from {}".format(mode, directory))
    
    X     = torch.load(os.path.join( directory, "X_{}.pt".format(mode) ) )
    y     = torch.load(os.path.join( directory, "y_{}.pt".format(mode) ) ) 
    paths = torch.load(os.path.join( directory, "paths_{}.pt".format(mode) ) )

    return (X, y, paths)

class EmergencyDataset(Dataset):
    
    def __init__(self, mode = "train"):
        
        assert mode in ["train", "val", "test"]
        self.X, self.y, self.paths = load_preprocessed_data_from_cache(mode = mode)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.X)
    
    def getPath(self, index):
        return self.paths[index]
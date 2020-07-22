import os
import torch
from torch.utils.data import Dataset
import config

'''
To create the .pt-files from scratch (i.e., run whole pre-processing pipeline: cropping/padding, normalizing, downsampling, etc.): run the file `lib dat dat`a PrepareData.py` in branch `pascal_lib`
'''

def load_preprocessed_data_from_cache(mode = "training"):
    
    directory = config.DATASET1_DATA_8K_DIR_OLD
    print("Loading cached {} data from {}".format(mode, directory))

    
    X     = torch.load(os.path.join( directory, "X_{}.pt".format(mode) ) )
    y     = torch.load(os.path.join( directory, "y_{}.pt".format(mode) ) ) 
    paths = torch.load(os.path.join( directory, "paths_{}.pt".format(mode) ) )

    return (X, y, paths)

class EmergencyDataset(Dataset):
    
    def __init__(self, mode = "train", split_mode="training"):
        print("DEPRECATED EM Dataset")
        # initially "mode" was used, thats deprecated now
        # now we use "split_mode"
        if split_mode == "training":
            mode = "train"
        elif split_mode == "validation":
            mode = "val"
        elif split_mode == "testing":
            mode = "test"

        assert mode in ["train", "val", "test"]
        self.X, self.y, self.paths = load_preprocessed_data_from_cache(mode = mode)
        
    def __getitem__(self, index):
        return {'audio': self.X[index], 'sample_rate': 8000, 'label': self.y[index]}
    
    def __len__(self):
        return len(self.X)
    
    def getPath(self, index):
        return self.paths[index]

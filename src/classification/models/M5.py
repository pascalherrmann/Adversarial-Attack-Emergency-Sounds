import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from classification.trainer.GeneralPLModule import GeneralPLModule
from datasets.EmergencyDataset import EmergencyDataset
from torch.utils.data import DataLoader

# helper-layer for reshaping
class PermuteLayer(nn.Module):
    def __init__(self, *args):
        super(PermuteLayer, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.permute(self.shape)
    
class M5(nn.Module):
    def __init__(self, hparams):
        super(M5, self).__init__()
        
        hparams.setdefault("p_drop", 0)

        self.datasets = {}
        self.model = nn.Sequential(
            nn.Conv1d(1, 128, 80, 4),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Dropout(p=hparams["p_drop"]),
            nn.Conv1d(128, 128, 3),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Dropout(p=hparams["p_drop"]),
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(4),
            nn.Dropout(p=hparams["p_drop"]),
            nn.Conv1d(256, 512, 3),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(4),
            nn.AvgPool1d(77),      #input should be 512x30 so this outputs a 512x1 # now: 64x512x77
            PermuteLayer(0, 2, 1), #change the 512x1 to 1x512
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        if type(x) == dict and "audio" in x:
            x  = x['audio'][:8000]
        if x.dtype == torch.float64:
            x = x.float()
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)  # if [batch_size, 8000] make it to [batch_size, 1, 8000]

        x = self.model(x)

        scores =  F.log_softmax(x, dim = 2)

        scores = scores.permute(1, 0, 2) # [1, 64, 2]

        scores = scores[0]
        return scores # this output should be of shape [BATCH_SIZE, 2]

class M5PLModule(GeneralPLModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = M5(hparams)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 8000}
        dataset_params = {'num_workers': 1, 'pin_memory': True} if self.device == 'cuda' else {}
        dataset_params['fixed_padding'] = True
        return dataset_type, dataset_params

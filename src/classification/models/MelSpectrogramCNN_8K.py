'''
hparams: = {'batch_size': 16, 
            'learning_rate': 0.0009471138112165006, 
            'p_dropout': 0.3394112556659779, 
            'n_hidden': 711, 
            'lr_decay': 0.7514824092200452, 
            'weight_decay': 0.003018912473366329}
-> spectrogramxxx_v359_acc_0.8998_best.p
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from classification.trainer.GeneralPLModule import GeneralPLModule

class MelSpectrogramCNN_8K(nn.Module):
    
    def __init__(self, hparams):
        super(MelSpectrogramCNN_8K, self).__init__()
        self.datasets = {}
        
        self.windowsize = 800
        self.window = torch.hann_window(self.windowsize).cuda()
        
        self.convs = nn.Sequential(
                    nn.BatchNorm2d(1),
                    nn.Conv2d(1, 10, kernel_size=10,stride=2),
                    nn.BatchNorm2d(10),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
            
                    nn.Conv2d(10, 20, kernel_size=10,stride=1),
                    nn.BatchNorm2d(20),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                        ## errorous
                    nn.Conv2d(20, 40, kernel_size=5,stride=1),
                    nn.BatchNorm2d(40),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                )
        
        self.dense = nn.Sequential(                 
                        nn.Linear(2400, hparams["n_hidden"]),
                        nn.PReLU(),
                        nn.Dropout(p=hparams["p_dropout"], inplace=False),
                        nn.Linear(hparams["n_hidden"], 2)  
                    )

    def forward(self, batch):
        x = batch['audio']
        x = torchaudio.transforms.MelSpectrogram().cuda()(x)
        # debug
        x = x.unsqueeze(1).float()
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return F.log_softmax(x,dim=1)

class MelSpectrogramCNN_8KPLModule(GeneralPLModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = MelSpectrogramCNN_8K(hparams)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 8000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params
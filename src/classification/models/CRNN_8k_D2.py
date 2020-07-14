import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchparse import parse_cfg
import numpy as np
import logging
import os

from classification.models.crnn.audio import MelspectrogramStretch
from classification.trainer.GeneralPLModule import GeneralPLModule

# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class CRNN8k_D2(nn.Module):
    def __init__(self, hparams, state_dict=None, device='cuda'):
        super(CRNN8k_D2, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = device
        
        self.in_chan = 1
        self.classes = ['negative', 'positive']
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                          sample_rate=8000, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])        
        
        self.spec = MelspectrogramStretch(sample_rate=8000)
        self.convs = nn.Sequential(
            
                    nn.Conv2d(1, 32, kernel_size=(3, 3)),
                    nn.BatchNorm2d(32),
                    nn.ELU(alpha=1.0),
            
                    nn.Conv2d(32, 32, kernel_size=(3, 3)),
                    nn.BatchNorm2d(32),
                    nn.ELU(alpha=1.0),
            
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"]),
            
            
                    nn.Conv2d(32, 64, kernel_size=(3, 3)),
                    nn.BatchNorm2d(64),
                    nn.ELU(alpha=1.0),
            
                    nn.Conv2d(64, 64, kernel_size=(3, 3)),
                    nn.BatchNorm2d(64),
                    nn.ELU(alpha=1.0),
            
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"]),
            
            
                    nn.Conv2d(64, 128, kernel_size=(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ELU(alpha=1.0),
            
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"]),
            
            
                    nn.Conv2d(128, 64, kernel_size=(3, 3)),
                    nn.BatchNorm2d(64),
                    nn.ELU(alpha=1.0),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"]),
                   )

        self.lstm_hidden_size= hparams["lstm_hidden_size"]
        self.LSTMCell = nn.LSTMCell(input_size=320, hidden_size=self.lstm_hidden_size)
        
        self.dense = nn.Sequential(
                        nn.Dropout(p=0.3, inplace=False),
                        nn.BatchNorm1d(self.lstm_hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Linear(in_features=self.lstm_hidden_size, out_features=2, bias=True)
                    )

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.convs.named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = ((lengths + 2*p - k)//s + 1).long()

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):
        x = batch['audio'].float()
        batch_size = x.size(0)
        #print(x.shape)
        
        # x-> (batch, time, channel)
        x = x.unsqueeze(2) # add channel dim
        
        # x-> (batch, channel, time)
        x = x.transpose(1,2)
        
        lengths = torch.tensor(x.shape[-1]).repeat(x.shape[0])
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)                
        
        # (batch, channel, freq, time)
        x = self.convs(x)
        lengths = self.modify_lengths(lengths)
        
        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        #print(x.shape)
        # x -> time, batch, data
        x = x.transpose(0,1)
        #print(x.shape)
        
        hx = torch.zeros(batch_size, self.lstm_hidden_size).to(self.device)
        cx = torch.zeros(batch_size, self.lstm_hidden_size).to(self.device)
        for i in range(x.size(0)):
            hx, cx = self.LSTMCell(x[i], (hx, cx))
        
        #print(hx.shape)
        
        # (batch, classes)
        x = self.dense(hx)
        
        #print(x.shape)
        return x
    
class CRNN8k_D2_PLModule(GeneralPLModule):
    
    def __init__(self, hparams, state_dict=None):
        super().__init__(hparams)
        self.model = CRNN8k_D2(hparams, state_dict=state_dict)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 8000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params
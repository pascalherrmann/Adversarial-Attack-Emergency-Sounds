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
class CRNN(nn.Module):
    def __init__(self, state_dict=None):
        super(CRNN, self).__init__()
        print("HI")
        self.logger = logging.getLogger(self.__class__.__name__)

        self.in_chan = 1
        self.classes = ['negative', 'positive']
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])        
        
        self.spec = MelspectrogramStretch()
        self.convs = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ELU(alpha=1.0),
                    nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False),
                    nn.Dropout(p=0.1, inplace=False),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ELU(alpha=1.0),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                    nn.Dropout(p=0.1, inplace=False),
                    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=[0, 0]),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ELU(alpha=1.0),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
                    nn.Dropout(p=0.1, inplace=False),
                   )
        self.recur = nn.LSTM(128, 64, num_layers=2, bidirectional=True)
        self.dense = nn.Sequential(
                        nn.Dropout(p=0.3, inplace=False),
                        nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Linear(in_features=128, out_features=2, bias=True)
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
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.recur(x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
    
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.dense(x)
        
        return x
    
class CRNNPLModule(GeneralPLModule):
    
    def __init__(self, hparams, state_dict=None):
        super().__init__(hparams)
        self.model = CRNN(state_dict)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 48000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params
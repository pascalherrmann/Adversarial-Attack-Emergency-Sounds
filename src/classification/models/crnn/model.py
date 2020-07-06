import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchparse import parse_cfg

import .audio
from classification.trainer.GeneralPLModule import GeneralPLModule

# Architecture inspiration from: https://github.com/keunwoochoi/music-auto_tagging-keras
class AudioCRNN(nn.Module):
    def __init__(self, config={}, state_dict=None):
        super(AudioCRNN, self).__init__()
        self.datasets = {}
        in_chan = 2 if config['transforms']['args']['channels'] == 'stereo' else 1

        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        
        self.classes = ['negative', 'positive']
        self.lstm_units = 64
        self.lstm_layers = 2
        self.spec = MelspectrogramStretch(hop_length=None, 
                                num_mels=128, 
                                fft_length=2048, 
                                norm='whiten', 
                                stretch_param=[0.4, 0.4])
        self.net = parse_cfg(config['cfg'], in_shape=[in_chan, self.spec.n_mels, 400])

    def _many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def modify_lengths(self, lengths):
        def safe_param(elem):
            return elem if isinstance(elem, int) else elem[0]
        
        for name, layer in self.net['convs'].named_children():
            #if name.startswith(('conv2d','maxpool2d')):
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                p, k, s = map(safe_param, [layer.padding, layer.kernel_size,layer.stride]) 
                lengths = ((lengths + 2*p - k)//s + 1).long()

        return torch.where(lengths > 0, lengths, torch.tensor(1, device=lengths.device))

    def forward(self, batch):
        
        x = batch.float()
        #x, lengths= batch['audio'], batch['lengths'] # unpacking seqs, lengths and srs
        
        # x-> (batch, time, channel)
        x = x.unsqueeze(2) # add channel dim
        
        # x-> (batch, channel, time)
        x = x.transpose(1,2)
        
        lengths = torch.tensor(x.shape[-1]).repeat(x.shape[0])
        # x -> (batch, channel, freq, time)
        x, lengths = self.spec(x, lengths)                
        
        # (batch, channel, freq, time)
        x = self.net['convs'](x)
        lengths = self.modify_lengths(lengths)
        
        # x -> (batch, time, freq, channel)
        x = x.transpose(1, -1)

        # xt -> (batch, time, channel*freq)
        batch, time = x.size()[:2]
        x = x.reshape(batch, time, -1)
        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
    
        # x -> (batch, time, lstm_out)
        x_pack, hidden = self.net['recur'](x_pack)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x_pack, batch_first=True)
    
        # (batch, lstm_out)
        x = self._many_to_one(x, lengths)
        # (batch, classes)
        x = self.net['dense'](x)
        
        return x
    
class AudioCRNNPLModule(GeneralPLModule):

    def __init__(self, hparams, config={}, state_dict=None):
        super().__init__(hparams)
        self.model = AudioCRNN(config, state_dict)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 48000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params
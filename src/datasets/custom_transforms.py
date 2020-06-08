# https://github.com/ksanjeevan/crnn-audio-classification/blob/master/data/transforms.py
import numpy as np
from torchvision import transforms

import torch


class ToTensorAudio(object):

    def __call__(self, data):
        """
        Input
            data[0]: audio signal in time, np array
            data[1]: sample rate
        """
        
        return torch.from_numpy(data[0]), data[1]

    def __repr__(self):
        return self.__class__.__name__ + '()'


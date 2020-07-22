import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from classification.trainer.GeneralPLModule import GeneralPLModule

from torchaudio.transforms import Spectrogram, MelSpectrogram , ComplexNorm, MelScale
from torchaudio.transforms import TimeStretch, AmplitudeToDB 
from torch.distributions import Uniform

class AblationModel(nn.Module):
    
    def __init__(self, hparams):
        super(AblationModel, self).__init__()
        self.datasets = {}
        
        self.random_time_stretch = hparams["random_time_stretch"]
        self.mel_scale = hparams["mel_scale"]
        self.normalize_spectrogram = hparams["normalize_spectrogram"]
        
        self.windowsize = 800
        self.window = torch.hann_window(self.windowsize).cuda()
        
        self.convs = nn.Sequential(
                    nn.BatchNorm2d(1),
                    nn.Conv2d(1, 10, kernel_size=10,stride=1),
                    nn.BatchNorm2d(10),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                    #Print(),
                    nn.Conv2d(10, 20, kernel_size=10,stride=1),
                    nn.BatchNorm2d(20),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                    #Print(),
                    nn.Conv2d(20, 40, kernel_size=10,stride=1),
                    nn.BatchNorm2d(40),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                    #Print(),
                    nn.Conv2d(40, 80, kernel_size=2,stride=1),
                    nn.BatchNorm2d(80),
                    nn.PReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout(p=hparams["p_dropout"], inplace=False),
                    #Print()
                )
        
        self.dense = nn.Sequential(                 
                        nn.Linear(80, hparams["n_hidden"]),
                        nn.PReLU(),
                        nn.Dropout(p=hparams["p_dropout"], inplace=False),
                        nn.Linear(hparams["n_hidden"], 2)  
                    )
        
        self.random_stretch = RandomTimeStretch(0.4)
        
        # Normalization (pot spec processing)
        self.complex_norm = ComplexNorm(power=2.)
        self.norm = SpecNormalization("whiten")

    def forward(self, batch):
        x = batch['audio']
        
        x = torchaudio.transforms.Spectrogram(power=None, normalized=False).cuda()(x)
        
        if self.training and self.random_time_stretch:
            x, _ = self.random_stretch(x)
        
        # we would do this usually in spectrogram (above: normalized=False)<-> no ablation needed
        x = self.complex_norm(x) 
    
        if self.mel_scale:
            x = MelScale().cuda()(x)
        
        if self.normalize_spectrogram:
            x = self.norm(x)

        x = x.unsqueeze(1).float()
        x = self.convs(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = F.avg_pool1d(x, kernel_size=x.size()[2:]).squeeze(2)
        x = self.dense(x)
        return F.log_softmax(x,dim=1)

class AblationModelPLModule(GeneralPLModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = AblationModel(hparams)
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 8000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params

class RandomTimeStretch(TimeStretch):

    def __init__(self, max_perc, hop_length=None, n_freq=201, fixed_rate=None):

        super(RandomTimeStretch, self).__init__(hop_length, n_freq, fixed_rate)
        self._dist = Uniform(1.-max_perc, 1+max_perc)

    def forward(self, x):
        rate = self._dist.sample().item()
        return super(RandomTimeStretch, self).forward(x, rate), rate

class SpecNormalization(nn.Module):

    def __init__(self, norm_type, top_db=80.0):

        super(SpecNormalization, self).__init__()

        if 'db' == norm_type:
            self._norm = AmplitudeToDB(stype='power', top_db=top_db)
        elif 'whiten' == norm_type:
            self._norm = lambda x: self.z_transform(x)
        else:
            self._norm = lambda x: x
    
    def z_transform(self, x):
        # Independent mean, std per batch
        non_batch_inds = [1, 2]
        mean = x.mean(non_batch_inds, keepdim=True)
        std = x.std(non_batch_inds, keepdim=True)
        x = (x - mean)/std 
        return x

    def forward(self, x):
        return self._norm(x)


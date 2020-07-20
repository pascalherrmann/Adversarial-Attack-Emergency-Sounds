import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio

from classification.trainer.GeneralPLModule import GeneralPLModule

class AblationModel(nn.Module):
    
    def __init__(self, hparams):
        super(AblationModel, self).__init__()
        self.datasets = {}
        
        self.normal_spectrogram = hparams["normal_spectrogram"]
        
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

    def forward(self, batch):
        x = batch['audio']
        if self.normal_spectrogram:
            x = torchaudio.transforms.Spectrogram().cuda()(x)
        else:
            x = torchaudio.transforms.MelSpectrogram().cuda()(x)

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
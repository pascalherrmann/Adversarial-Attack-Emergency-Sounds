import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classification.trainer.GeneralPLModule import GeneralPLModule

class DeepRecursiveCNN8k(nn.Module):
    def __init__(self, p_drop=0.2, hidden_dim=100, numChunksList=[5,2,1]):
        super(DeepRecursiveCNN8k, self).__init__()
        self.datasets={}
        
        self.hidden_dim = hidden_dim 
        
        self.numChunksList = numChunksList # full sequence, half/half, 2sec split
            
        # initial normalization
        self.bn0 = nn.BatchNorm1d(1)
        
        ## init M5-mod with global pooling
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(p=p_drop)
        
        self.conv2 = nn.Conv1d(128, 256, 3)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(256, 512, 3)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(512, 1024, 3)
        self.bn4 = nn.BatchNorm1d(1024)
        self.pool4 = nn.MaxPool1d(4)
        
        self.fc1 = nn.Linear(1024, self.hidden_dim)

        # init final FC
        self.fcN = nn.Linear(sum(self.numChunksList) * hidden_dim, 2)
    
    def forwardM5(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.drop1(self.pool1(x))
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, kernel_size=x.size()[2:]) # global average pooling
        x = x.squeeze(2)
        x = self.fc1(x)
        return x
    
    def forward(self, batch):
        x0 = batch['audio']
        batch_size = x0.shape[0]
        
        ## first normalize batch
        x0 = x0.unsqueeze(1)
        x0 = self.bn0(x0)
        
        cnnResult = []
        for numChunks in self.numChunksList:
            chunks = torch.chunk(x0, numChunks, dim=2)
            
            for chunk in chunks:
                x = self.forwardM5(chunk) ## do one convolution
                x = x.unsqueeze(1)
                cnnResult.append(x)
        
        x = torch.cat(cnnResult, dim=2)
        x = x.squeeze(1)

        # final linear layer
        x = self.fcN(F.relu(x))

        return F.log_softmax(x,dim=1)

class DeepRecursiveCNN8kPLModule(GeneralPLModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = DeepRecursiveCNN8k(p_drop=hparams['p_drop'], hidden_dim=hparams['n_hidden'])
        
    def dataset_info(self):
        dataset_type = {"sample_rate": 8000}
        dataset_params = {"fixed_padding": True}
        return dataset_type, dataset_params
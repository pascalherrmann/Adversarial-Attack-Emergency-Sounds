import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        
        self.windowsize = 2048
        self.window = torch.hann_window(self.windowsize).cuda()
        
        self.bn0 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 5, kernel_size=10,stride=2)
        self.bn1 = nn.BatchNorm2d(5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=10,stride=2)
        self.bn2 = nn.BatchNorm2d(5)
        self.conv3 = nn.Conv2d(5, 10, kernel_size=20,stride=3)
        self.bn3 = nn.BatchNorm2d(10)
        self.conv4 = nn.Conv2d(10, 15, kernel_size=20,stride=3)
        self.bn4 = nn.BatchNorm2d(15)
        self.fc1 = nn.Linear(5100, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, batch):
        x = batch['audio']
        x = torch.stft(x, self.windowsize, window=self.window).pow(2).sum(3).sqrt()
        x = x.unsqueeze(1).float()
        x = self.bn0(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 5100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x,dim=1)
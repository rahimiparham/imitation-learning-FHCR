#build cnn model

import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 128 -> 32 -> 16 -> 8
        self.conv = nn.Sequential(
            nn.Conv2d(3,24,3,1,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(24,48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48,96,3,1,1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fcn_goal = nn.Sequential(
            nn.Linear((8*8*96)*5 + 7,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )
        # #convolutional layer
        # self.conv1 = 
        # self.batch_norm1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16,64,3,1,1)
        # self.batch_norm2 = nn.BatchNorm2d(64)
        # self.conv3= nn.Conv2d(64,256,3,1,1)
        # self.batch_norm3 = nn.BatchNorm2d(256)
        # #fully connected layer
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(5*256*5,128)
        # self.fc2 = nn.Linear(128,7)
        
    def forward(self,x1, x2):
        x1 = torch.concat([self.conv(x1[:,i,:,:,:]) for i in range(5)], dim=1)
        output = self.fcn_goal(torch.concat([x1,x2],dim=1))
        return output
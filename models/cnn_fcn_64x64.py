#build cnn model

import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 64 -> 32 -> 16
        self.conv = nn.Sequential(
            nn.Conv2d(3,24,3,1,1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24,48,3,1,1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48,96,3,1,1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fcn_goal = nn.Sequential(
            nn.Linear((16*16*96)*5 + 7,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )
        
    def forward(self,x1, x2):
        y1 = self.conv(x1[:,0,:,:,:])
        y2 = self.conv(x1[:,1,:,:,:])
        y3 = self.conv(x1[:,2,:,:,:])
        y4 = self.conv(x1[:,3,:,:,:])
        y5 = self.conv(x1[:,4,:,:,:])
        x1 = torch.concat((y1, y2, y3, y4, y5), dim=1)
        x1 = torch.concat((x1,x2),dim=1)
        return self.fcn_goal(x1)
#build cnn model

import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # 128 -> 32 -> 16
        self.conv = nn.Sequential(
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(15,40,3,1,1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(40,80,3,1,1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ConstantPad2d(1, 0),
            nn.Conv2d(80,160,3,1,1),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fcn_goal = nn.Sequential(
            nn.Flatten(),
            nn.Linear((9*9*160),64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
        self.fcn_control = nn.Sequential(
            nn.Linear(3 + 7,32),
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
        x1 = self.conv(x1)
        # print("conv done!", x1.shape)
        x1 = self.fcn_goal(x1)
        # print(x1.shape,x2.shape)
        z = torch.concat([x1,x2],dim=1)
        # print(z.shape)
        output = self.fcn_control(z)
      
        return output
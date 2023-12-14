import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        alexnet = models.alexnet(pretrained=True)
        self.encoder = alexnet.features
        # 128 -> 32 -> 16 -> 8
        self.fcn = nn.Sequential(
            nn.Linear((2304*5) + 7,96),
            nn.ReLU(),
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )
        self.flattener = nn.Flatten()
        
    def forward(self,x1, x2):
        # print('concat')
        y1 = self.encoder(x1[:,0,:,:,:])
        y2 = self.encoder(x1[:,1,:,:,:])
        y3 = self.encoder(x1[:,2,:,:,:])
        y4 = self.encoder(x1[:,3,:,:,:])
        y5 = self.encoder(x1[:,4,:,:,:])
        y1 = self.flattener(y1)
        y2 = self.flattener(y2)
        y3 = self.flattener(y3)
        y4 = self.flattener(y4)
        y5 = self.flattener(y5)
        x1 = torch.concat((y1, y2, y3, y4, y5, x2), dim=1)
        return self.fcn(x1)
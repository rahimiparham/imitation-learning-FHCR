"""
This file is used to calculate and display test dataset loss for a trained NN.
The test dataset is not used during training or hyperparameter tuning
"""

import torch.nn as nn
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset

from models.cnn_fcn_64x64 import Net
dataset_path = "dataset/combined/dataset_downsampled.pkl"
saved_model_path = 'cnn_fcn_64x64.pth'

def calc_test_loss(model, test_dataloader, loss_fn):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, positions, labels = data
            images = images.to(device)
            images = images.to(torch.float32)
            labels = (labels.to(device))
            positions = (positions.to(device))
            outputs = model(x1=images, x2=positions)
            loss += loss_fn(outputs, labels).item()

    return(loss/len(test_dataloader))

with (open(dataset_path, "rb")) as pickle_file:
    dataset = pickle.load(pickle_file)

dataset["test"]["rgbs"] = np.moveaxis(dataset["test"]["rgbs"], 4, 2)
num_test_samples = dataset["test"]["rgbs"].shape[0]
print("testing with {} samples".format(num_test_samples))

test_dataset = TensorDataset(torch.tensor(dataset["test"]["rgbs"], dtype=torch.uint8)
                              , torch.Tensor(dataset["test"]["joint_positions"])
                              , torch.Tensor(dataset["test"]["joint_velocities"]))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                        shuffle=False, num_workers=0)

model = Net()
print("number of model params:", sum(p.numel() for p in model.parameters()))
model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
loss_fn = nn.MSELoss()
avg_loss = calc_test_loss(model, test_dataloader, loss_fn)

print("average test loss for network: ", avg_loss)

import torch.nn as nn
import pickle
from models.cnn_fcn2 import Net
import torch
import numpy as np
from torch.utils.data import TensorDataset

from torch.optim import Adam

from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

dataset_path = "dataset/combined/dataset.pkl"

def saveModel(model):
    path = "cnn_fcn2.pth"
    torch.save(model.state_dict(), path)

def calc_val_loss(model, val_dataloader, loss_fn):
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for data in val_dataloader:
            images, positions, labels = data
            outputs = model(x1=images, x2=positions)
            loss += loss_fn(outputs, labels).item()
    
    return(loss/len(val_dataloader))


def train(num_epochs, model, train_dataloader, val_dataloader, optimizer, loss_fn, patience = 7):
    
    best_loss = 1e8

    last_best_epoch = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    model.to(device)

    train_loss_list = []
    val_loss_list = []
    for epoch in tqdm(range(num_epochs)): 
        running_loss = 0.0
        for i, (images, positions, labels) in enumerate(train_dataloader, 0):
            
            images = (images.to(device))
            labels = (labels.to(device))
            optimizer.zero_grad()
            outputs = model(x1= images, x2= positions)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()    

        print('n', len(train_dataloader))
        running_loss /= len(train_dataloader)
        train_loss_list.append(running_loss)
        print('For epoch {} the training loss is {}'.format(epoch+1, running_loss))
        running_loss = 0.0

        val_loss = calc_val_loss(model, val_dataloader, loss_fn)
        print('For epoch {} the validation loss is {}'.format(epoch+1, val_loss))
        val_loss_list.append(val_loss)
        if val_loss < best_loss:
            saveModel(model)
            best_loss = val_loss
            last_best_epoch = epoch

        if last_best_epoch and epoch-last_best_epoch>patience:
            print('stopping training because of no improvement after {} epochs in epoch {}'.format(patience, epoch+1))
            break

    return train_loss_list, val_loss_list

def main():
    
    model = Net()
    print("number of model params:", sum(p.numel() for p in model.parameters()))

    with (open(dataset_path, "rb")) as pickle_file:
        dataset = pickle.load(pickle_file)
    dataset["train"]["rgbs"] = np.moveaxis(dataset["train"]["rgbs"], 4, 2)
    dataset["val"]["rgbs"] = np.moveaxis(dataset["val"]["rgbs"], 4, 2)

    num_train_samples = dataset["train"]["rgbs"].shape[0]
    num_val_samples = dataset["val"]["rgbs"].shape[0]

    print("training with {} training samples and {} validation samples".format(num_train_samples, num_val_samples))

    train_dataset = TensorDataset(torch.Tensor(dataset["train"]["rgbs"])
                                  , torch.Tensor(dataset["train"]["joint_positions"])
                                  , torch.Tensor(dataset["train"]["joint_velocities"]))
    val_dataset = TensorDataset(torch.Tensor(dataset["val"]["rgbs"])
                                  , torch.Tensor(dataset["val"]["joint_positions"])
                                  , torch.Tensor(dataset["val"]["joint_velocities"]))

    del dataset

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, 
                                            shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=num_val_samples, 
                                            shuffle=False, num_workers=0)
    
    del train_dataset
    del val_dataset
    
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_loss_list, val_loss_list = train(70, model, train_dataloader, val_dataloader, optimizer, loss_fn)

    fig1 = plt.figure("train")
    plt.plot(train_loss_list, c="red")
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    fig2 = plt.figure("val")
    plt.plot(val_loss_list, c="green")
    plt.title("Validation Loss")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()

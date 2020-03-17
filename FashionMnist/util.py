import torch
import torchvision
from torchvision import transforms

import os
import numpy as np
from variables import*
from matplotlib import pyplot as plt


torch.set_printoptions(linewidth=120)

def get_data():
    absolute_train_path = os.path.join(os.getcwd(),train_path)
    absolute_test_path  = os.path.join(os.getcwd(),test_path)
    if os.path.exists(absolute_train_path):
        print("Train data loading !!!")
        train_set = torchvision.datasets.FashionMNIST(
                    root = absolute_train_path,
                    train=True,
                    download=False,
                    transform = transforms.Compose([
                        transforms.ToTensor()
                    ]))

    else:
        print("Train data downloading !!!")
        train_set = torchvision.datasets.FashionMNIST(
                    root = absolute_train_path,
                    train=True,
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))

    if os.path.exists(absolute_test_path):
        print("Test data loading !!!")
        test_set = torchvision.datasets.FashionMNIST(
                    root = absolute_test_path,
                    train=False,
                    download=False,
                    transform = transforms.Compose([
                        transforms.ToTensor()
                    ]))

    else:
        print("Test data downloading !!!")
        test_set = torchvision.datasets.FashionMNIST(
                    root = absolute_test_path,
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor()
                    ]))

    train_loader = torch.utils.data.DataLoader(
                    train_set,
                    batch_size=batch_size,
                    shuffle=True
                    )

    test_loader = torch.utils.data.DataLoader(
                    test_set,
                    batch_size=batch_size,
                    shuffle=True
                    )
    plot_batch(train_loader)
    plot_batch(test_loader)

    return train_loader, test_loader

def plot_batch(loader, is_show=False):
    if is_show:
        batch = next(iter(loader))
        images, labels = batch
        grid = torchvision.utils.make_grid(images, nrow=len(labels))

        plt.figure(figsize=(15,15))
        plt.imshow(np.transpose(grid, (1,2,0)))
        plt.show()
        print('labels :',labels)
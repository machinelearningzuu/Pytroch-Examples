import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from time import time
from variables import*
from util import get_data

class FashionMnistClassifier(nn.Module):
    def __init__(self):
        super(FashionMnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=ofm1,
                                kernel_size=kernal_size
                                )
        ofm1_dim = input_size - kernal_size + 1
        ofm1_dim = ((ofm1_dim - kernal_pool)// stride) + 1

        self.conv2 = nn.Conv2d(
                                in_channels=ofm1,
                                out_channels=ofm2,
                                kernel_size=kernal_size
                                )
        ofm2_dim = ofm1_dim - kernal_size + 1
        ofm2_dim = ((ofm2_dim - kernal_pool)// stride) + 1
        self.flatten_dim = ofm2 * ofm2_dim * ofm2_dim

        self.fc1   =  nn.Linear(
                                in_features=self.flatten_dim,
                                out_features=dense1
                                )
        self.fc2   =  nn.Linear(
                                in_features=dense1,
                                out_features=dense2
                                )
        self.fc3   =  nn.Linear(
                                in_features=dense2,
                                out_features=output
                                )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernal_pool, stride=stride)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernal_pool, stride=stride)

        x = F.relu(self.fc1(x.reshape(-1,self.flatten_dim)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)  , we dnt use softmax because the loss function implemnt softmax itself
        return x

class Model(object):
    def __init__(self, device, model):
        train_loader, test_loader = get_data()

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader  = test_loader

    def train_model(self):
        loss_function = F.cross_entropy
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        t0 = time()
        train_loss = []
        for epoch in range(num_epochs):
            epoch_loss = 0
            n_correct = 0
            for batch in self.train_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)

                loss = loss_function(preds, labels)

                optimizer.zero_grad() # avoid accumilating gradients
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                epoch_loss += loss.item()
                n_correct += preds.argmax(dim=1).eq(labels).sum().item()

            train_accuracy = round(n_correct / len(self.train_loader),3)
            print("epoch :",epoch," train_loss :",round(epoch_loss,3)," train_accuracy :",train_accuracy,"%")

        t1 = time()
        print("train time with {}: {}".format(self.device,t1-t0))

        plt.plot(np.array(train_loss))
        plt.show()

    def prediction(self):
        with torch.no_grad():
            n_correct = 0
            for batch in self.test_loader:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                n_correct += preds.argmax(dim=1).eq(labels).sum().item()
        print("Val_accuracy: {}".format(n_correct / len(self.test_loader) /batch_size))

if __name__ == "__main__":
    torch.set_grad_enabled(True)

    is_model_heavy = True
    device = torch.device('cuda:0') if is_model_heavy and torch.cuda.is_available() else torch.device('cpu')
    print("Running on {}".format(device))

    model = FashionMnistClassifier()
    model = model.to(device)
    classifier = Model(device,model)
    classifier.train_model()
    classifier.prediction()
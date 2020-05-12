import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from util import get_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam

class RegressionNN(object):
    def __init__(self, train_data, test_data):
        self.train_data = DataLoader(
                            train_data,
                            batch_size=32,
                            shuffle=True
                            )

        self.train_data = DataLoader(
                            train_data,
                            batch_size=32,
                            shuffle=True
                            )

    def regressor(self):
        return nn.Linear(
                    in_features=5,
                    out_features=1
                    )

    def loss_fnc(self, Ypred, Y):
        return F.mse_loss(Ypred, Y)

    def optimizer(self, learning_rate=0.1):
        return Adam(self.regressor().parameters(), lr=learning_rate)

    def train(self, num_epochs=100):
        model = self.regressor()
        opt = self.optimizer()
        for i in range(1,num_epochs+1):
            train_loss = 0
            for X, Y in self.train_data:
                Ypred = model(X)
                loss = self.loss_fnc(Ypred, Y)
                loss.backward() # calculate gradients

                opt.step() # update parameters using claculated gradients
                opt.zero_grad() # use to avoid accumilating the gradients
                train_loss += loss.item()

            with torch.no_grad():
                test_loss = 0
                for X, Y in self.train_data:
                    Ypred = model(X)
                    loss = self.loss_fnc(Ypred, Y)
                    test_loss += loss.item()
            if i % 10 == 0:
                print("Train loss : {} Test loss : {}".format(train_loss, test_loss))

if __name__ == "__main__":
    train_data, test_data = get_data()
    model = RegressionNN(train_data, test_data)
    model.train()

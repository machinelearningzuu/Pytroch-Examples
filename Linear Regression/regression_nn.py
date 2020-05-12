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

    def optimizer(self, learning_rate=0.01):
        return Adam(self.regressor().parameters(), lr=learning_rate)

    def train(self, num_epochs=30):
        model = self.regressor()
        opt = self.optimizer()
        for _ in range(num_epochs):
            train_loss = 0
            for X, Y in self.train_data:
                Ypred = model(X)
                loss = self.loss_fnc(Ypred, Y)
                loss.backward()

                opt.step()
                opt.zero_grad() # use to avoid accumilating the gradients
                train_loss += loss.item()

            with torch.no_grad():
                test_loss = 0
                for X, Y in self.train_data:
                    Ypred = model(X)
                    loss = self.loss_fnc(Ypred, Y)
                    test_loss += loss.item()

            print("Train loss : {} Test loss : {}".format(train_loss, test_loss))

if __name__ == "__main__":
    train_data, test_data = get_data()
    model = RegressionNN(train_data, test_data)
    model.train()

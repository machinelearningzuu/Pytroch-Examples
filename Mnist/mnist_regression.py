import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from util import get_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import Adam
from variables import*
from matplotlib import pyplot as plt

class MnistRegression(object):
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = self.MnistModel()

    class MnistModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(
                    in_features=input_shape,
                    out_features=output_shape
                    )

        def forward(self, x):
            x = x.reshape(-1, input_shape)
            x = self.linear(x)
            x = F.log_softmax(x, dim=1)
            return x

    def loss_fnc(self, Ypred, Y):
        return F.cross_entropy(Ypred, Y)

    def optimizer(self, learning_rate=0.1):
        return Adam(self.model.parameters(), lr=learning_rate)

    def evaluate(self, Y, Ypred):
        P = torch.argmax(Ypred, dim=1).numpy()
        Y = Y.numpy()
        return np.sum(Y == P)

    def train(self, num_epochs=100):
        opt = self.optimizer()
        total_train_loss = []
        total_test_loss = []
        for i in range(1,num_epochs+1):
            n_correct = 0
            n_total = 0
            for X, Y in self.train_data:
                Y = Y.to(dtype=torch.int64)
                Ypred = self.model(X)
                loss = self.loss_fnc(Ypred, Y)
                loss.backward() # calculate gradients
                total_train_loss.append(loss.item())

                n_correct += self.evaluate(Y, Ypred)
                n_total += batch_size

                opt.step() # update parameters using claculated gradients
                opt.zero_grad() # use to avoid accumilating the gradients

            train_acc = round(n_correct/n_total, 3)

            with torch.no_grad():
                n_correct = 0
                n_total = 0
                for X, Y in self.test_data:
                    Y = Y.to(dtype=torch.int64)
                    Ypred = self.model(X)
                    loss = self.loss_fnc(Ypred, Y)
                    total_test_loss.append(loss.item())

                    n_correct += self.evaluate(Y, Ypred)
                    n_total += batch_size

                test_acc = round(n_correct/n_total, 3)

            print("Train Acc : {} Test Acc : {}".format(train_acc, test_acc))

        plt.plot(total_train_loss, label='Train loss')
        plt.plot(total_test_loss , label='Test loss')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    train_data, test_data = get_data()
    model = MnistRegression(train_data, test_data)
    model.train()

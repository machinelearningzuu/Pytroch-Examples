import torch
from torch.autograd import Variable
import numpy as np
from util import get_data

class LinearRegression(object):
    def __init__(self, Xtrain, Xtest, Ytrain, Ytest):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest

        self.w = torch.randn(5,1, requires_grad = True)
        self.b = torch.randn(1, requires_grad = True)

        print('Input data shape : {}'.format(Xtrain.shape))
        print('Output data shape : {}'.format(Ytrain.shape))

    def regressor(self, X):
        Ypred = torch.mm(X ,self.w) + self.b
        return Ypred.view(-1)

    def loss_fnc(self, Y, Ypred):
        diff = Y - Ypred
        loss = torch.sum(diff * diff) / diff.numel()
        # loss = Variable(loss.data, requires_grad=True)
        return loss

    def train(self, num_epochs=30, learning_rate = 0.01):
        for i in range(num_epochs):
            Ypred= self.regressor(self.Xtrain)
            loss = self.loss_fnc(self.Ytrain, Ypred)

            loss.backward()
            with torch.no_grad():
                self.w -= learning_rate * self.w.grad
                self.b -= learning_rate * self.b.grad

                self.w.grad.zero_()
                self.b.grad.zero_()

            Ypred= self.regressor(self.Xtest)
            val_loss = self.loss_fnc(self.Ytest, Ypred)

            train_loss = loss.item()
            test_loss = val_loss.item()
            print("Train loss : {}, Test loss : {}".format(train_loss, test_loss))

    def predict(self, X):
        return self.regressor(X)


if __name__ == "__main__":
    Xtrain, Xtest, Ytrain, Ytest = get_data()
    model = LinearRegression(Xtrain, Xtest, Ytrain, Ytest)
    model.train()
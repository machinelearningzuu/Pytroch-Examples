import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time
import os
import numpy as np
from time import time
from matplotlib import pyplot as plt

from variables import input_size, hidden1, hidden2, output, batch_size, num_epochs, learning_rate, label_encode, model_path
from util import get_data

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

class IrisClassifier(object):
    def __init__(self):
        Xtrain, Xtest, Ytrain, Ytest = get_data()

        train_data = TensorDataset(torch.FloatTensor(Xtrain), torch.tensor(Ytrain, dtype=torch.int64))
        test_data = TensorDataset(torch.FloatTensor(Xtest), torch.tensor(Ytest, dtype=torch.int64))

        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    def train(self, device, model):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_loss = []
        t0 = time()
        for epoch in range(num_epochs):
            epoch_loss = 0
            for instance, label in self.test_loader:
                model.zero_grad()
                output_tensor = model(instance.to(device))
                target = label.to(device).squeeze()
                loss = loss_function(output_tensor, target)
                epoch_loss += loss
                loss.backward()
                optimizer.step()
            train_loss.append(epoch_loss)
        t1 = time()
        print("Train time: ", (t1 - t0))

        plt.plot(np.array(train_loss))
        plt.show()

    def prediction(self, label_encode):
        with torch.no_grad():
            total_samples = 0
            n_correct = 0
            Predictions = []
            for instance, label in self.test_loader:
                output_tensor = model(instance.float().to(device))
                target = label.to(device).squeeze()
                P = torch.argmax(output_tensor, dim=1).squeeze()
                Predictions.extend([list(label_encode.keys())[i] for i in P.tolist()])
                n_correct += (P == target).float().sum()
                total_samples += len(P)
        print("Predictions : {}".format(Predictions))
        print("Val_accuracy: {}".format(n_correct / total_samples))

if __name__ == "__main__":
    is_model_heavy = True  # this is because for small models CPU perform better than GPU
    device = torch.device('cuda:0') if is_model_heavy and torch.cuda.is_available() else torch.device('cpu')
    print("Running on {}".format(device))

    model = LinearModel()
    model = model.to(device)
    if not os.path.exists(model_path):
        print("Creating the Model")
        torch.save(model.state_dict(), model_path)
    else:
        print("Model is existing")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    classifier = IrisClassifier()
    classifier.train(device, model)
    classifier.prediction(label_encode)

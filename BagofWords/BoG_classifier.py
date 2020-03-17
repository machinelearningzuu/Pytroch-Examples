import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim

import os
import numpy as np
from time import time
from matplotlib import pyplot as plt

from variables import hidden, output, num_epochs, learning_rate, encode_dict
from util import sample_data, word_embedding

class BoWClassifier(nn.Module):  # inheriting from nn.Module!
    def __init__(self, hidden, vocab_size, output):
        super(BoWClassifier, self).__init__()

        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, bow_vec):
        x = F.relu(self.fc1(bow_vec))
        return torch.sigmoid(self.fc2(x))

def train(device, model):
    word2idx,train_data,test_data = sample_data()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = []
    t0 = time()
    for epoch in range(num_epochs):
        epoch_loss = []
        for instance, label in train_data:
            model.zero_grad()
            bow_vec = word_embedding(instance, word2idx).float().to(device)
            output_tensor = model(bow_vec)
            target = torch.LongTensor([label]).to(device).float()
            loss = loss_function(output_tensor, target)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        train_loss.append(np.mean(epoch_loss))
    t1 = time()
    print("Train time: ",(t1-t0))

    plt.plot(np.array(train_loss))
    plt.show()

def predict():
    with torch.no_grad():
        n_correct = 0
        total_samples = 0
        Predictions = []
        for instance, label in test_data:
            bow_vec = word_embedding(instance, word2idx).float().to(device)
            output_tensor = model(bow_vec).squeeze().item()
            P = output_tensor>0.5
            target = torch.LongTensor([label]).to(device).float()
            n_correct += (P == target).float().sum()
            Predictions.append(list(encode_dict.keys())[P])
            total_samples += 1
        print("Predictions : {}".format(Predictions))
        print("Val_accuracy: {}".format(n_correct / total_samples))


if __name__ == "__main__":
    is_model_heavy = False # this is because for small models CPU perform better than GPU
    device = torch.device('cuda:0') if is_model_heavy and torch.cuda.is_available() else torch.device('cpu')
    print("Running on {}".format(device))

    word2idx,train_data,test_data = sample_data()
    model = BoWClassifier(hidden, len(word2idx), output).to(device)
    train(device, model)
    predict()



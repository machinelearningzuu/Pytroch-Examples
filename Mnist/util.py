import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from variables import*

def get_data():
    df = pd.read_csv(data_path)
    cols = df.columns.values
    Y = df[cols[0]].values
    X = df[cols[1:]].values/255.0

    X, Y = shuffle(X, Y)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                    X, Y,
                                    test_size=0.2,
                                    random_state=42
                                    )

    Xtrain = torch.Tensor(Xtrain)
    Ytrain = torch.Tensor(Ytrain)
    Xtest  = torch.Tensor(Xtest)
    Ytest  = torch.Tensor(Ytest)

    train_tensor = TensorDataset(Xtrain, Ytrain)
    test_tensor = TensorDataset(Xtest, Ytest)

    train_data = DataLoader(
                        train_tensor,
                        batch_size=batch_size,
                        shuffle=True
                        )

    test_data = DataLoader(
                        test_tensor,
                        batch_size=batch_size,
                        shuffle=True
                        )

    return train_data, test_data

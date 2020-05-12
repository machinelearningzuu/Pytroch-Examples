import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv('Fish.csv')
    colmns = df.columns.values
    Inputs = df[colmns[2:]].values
    labels = df[colmns[1]].values

    scalar = StandardScaler()
    X = scalar.fit_transform(Inputs)

    Ymean = labels.mean()
    Ystd = labels.std()

    Y = (labels - Ymean) / Ystd

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                                    X, Y,
                                    test_size=0.2,
                                    random_state=42
                                    )
    Xtrain = torch.Tensor(Xtrain)
    Ytrain = torch.Tensor(Ytrain)
    Xtest  = torch.Tensor(Xtest)
    Ytest  = torch.Tensor(Ytest)

    return Xtrain, Xtest, Ytrain, Ytest
import os
import numpy as np
import pandas as pd
from variables import csv_path, label_encode, file_name, cutoff
from sklearn.utils import shuffle

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.copy()
    df.dropna(axis=0, how='any', inplace=False)
    df['label'] = df.apply(y2indicator, axis=1)
    del df['species']
    df = shuffle(df)
    df.to_csv(file_name, encoding='utf-8')

def y2indicator(x):
    species = x['species']
    return label_encode[species]

def get_data():
    if not os.path.exists(file_name):
        preprocess_data(csv_path)
    df = pd.read_csv(file_name)
    Xdata = df.copy()[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
    Ydata = df.copy()[['label']].to_numpy()
    train_set = int(cutoff * len(df))
    Xtrain, Xtest = Xdata[:train_set], Xdata[train_set:]
    Ytrain, Ytest = Ydata[:train_set], Ydata[train_set:]
    return Xtrain, Xtest, Ytrain, Ytest

def one_hot_encode(Ydata):
    N = len(Ydata)
    num_classes = 3
    y = np.eye(num_classes)
    return y[Ydata]

from collections import Counter
import torch
from variables import encode_dict

def sample_data():
    train_data = [  ("me gusta comer en la cafeteria", "SPANISH"),
                    ("I love him a lot", "ENGLISH"),
                    ("corona sucks the world", "ENGLISH"),
                    ("Give it to me", "ENGLISH"),
                    ("No creo que sea una buena idea", "SPANISH"),
                    ("No it is not a good idea to get lost at sea", "ENGLISH")
                 ]

    test_data = [("Yo creo que si", "SPANISH"),
                ("it is lost on me", "ENGLISH")]

    train_data = tokenize_data(train_data)
    test_data = tokenize_data(test_data)
    word2idx = word2index(train_data,test_data)

    return word2idx,train_data,test_data

def tokenize_data(data):
    return [(s.split(' '),encode_dict[t]) for s,t in data]

def word2index(train_data,test_data):
    wordcount = []
    for text,_ in train_data:
        wordcount.extend(text)
    for text,_ in test_data:
        wordcount.extend(text)

    word2count = Counter(wordcount).most_common()
    word2idx = {w:i for i,(w,c) in enumerate(word2count)}
    return word2idx

def word_embedding(sentence, word2idx):
    vector = torch.zeros(len(word2idx))
    for word in sentence:
        vector[word2idx[word]] += 1
    return vector.view(1, -1)

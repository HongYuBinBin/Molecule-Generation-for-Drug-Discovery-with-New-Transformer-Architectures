import os
from io import open
import pandas as pd
import numpy as np
import pickle
import random

import torch
from torch.nn.utils.rnn import pad_sequence

with open('./new_dict.pickle', 'rb') as f:
    new_dict = pickle.load(f)

class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []

    def add_word(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2char)
    
    def show_val(self):
        return self.idx2char

my_dict = Dictionary()
for i in range(len(new_dict)):
    my_dict.add_word(new_dict[i])

class Corpus(object):
    def __init__(self, path):
        self.dictionary = my_dict
        self.vocab_size = 0
        
        self.data = self.tokenize(os.path.join(path))
        
    def tokenize(self, path):
        assert os.path.exists(path) 
 
        data = pd.read_csv(path) 
        
        # make array
        arr = np.array([]) 
        arr = np.append(arr, data) 
        
        # Tokenize file content
        idss = [] 
        for line in arr:
            seq = '$' + line + '!' 
            ids = [] 
            for char in seq:  
                ids.append(self.dictionary.char2idx[char])
            idss.append(torch.tensor(ids).type(torch.int64))

        self.vocab_size = len(self.dictionary)
        print(self.dictionary.show_val())
        
        return idss

def build_data(encoded):
    input_seq, label_seq = encoded[:-1], encoded[1:]
    input_seq = torch.LongTensor(input_seq)  
    label_seq = torch.LongTensor(label_seq) 
    return input_seq, label_seq

data = Corpus("../../../data/latent_covid19.csv")
# print("1.1 Add padding")

# Batchfy
vocab_size = data.vocab_size
mydata = data.data

# print("Before shuffle", mydata[0])
random.seed(7)
random.shuffle(mydata)
# print("After shuffle", mydata[0])

val_idx = int(len(mydata)*0.9)
train, validation= mydata[:val_idx], mydata[val_idx:]

X_t_train = []
Y_t_train = []
X_t_val = []
Y_t_val = []

for seq in train:
    X, Y = build_data(seq)
    X_t_train.append(X)
    Y_t_train.append(Y)

for seq in validation:
    X, Y = build_data(seq)
    X_t_val.append(X)
    Y_t_val.append(Y)

# Padding
# print("Make dataset and dataloader")
padded_sequence_x_train = pad_sequence(X_t_train, batch_first=True)
padded_sequence_y_train = pad_sequence(Y_t_train, batch_first=True)
padded_sequence_x_val = pad_sequence(X_t_val, batch_first=True)
padded_sequence_y_val = pad_sequence(Y_t_val, batch_first=True)

x_train_data = np.array(padded_sequence_x_train)
y_train_data = np.array(padded_sequence_y_train)
x_val_data = np.array(padded_sequence_x_val)
y_val_data = np.array(padded_sequence_y_val)

with open('x_train_latent_data.pickle', 'wb') as f:
    pickle.dump(x_train_data, f)

with open('x_val_latent_data.pickle', 'wb') as f:
    pickle.dump(x_val_data, f)

with open('y_train_latent_data.pickle', 'wb') as f:
    pickle.dump(y_train_data, f)

with open('y_val_latent_data.pickle', 'wb') as f:
    pickle.dump(y_val_data, f)

with open('new_dict_size.pickle', 'wb') as f:
    print("len dict", len(data.dictionary))
    pickle.dump(len(data.dictionary), f)

import neptune
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.utils.rnn as rnn_utils

import pandas as pd
import numpy as np

import torch.onnx  # save format

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#from Data import Dictionary

import os
from io import open

from Transformer_model import Encoder, Decoder, Transformer

# coding: utf-8
import argparse
import time
import math
import os
import torch.onnx

import pickle

from config import get_parser
from config import get_config

get_parser()
config = get_config()

cuda = False
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())
    print(torch.cuda.get_device_name(device))
    print(torch.cuda.is_available())
    cuda = True
    # True
else :
    print("NOT GPU")
    cuda = False

with open('./Data/x_train_data_ref.pickle', 'rb') as f:
    x_train_data = pickle.load(f)

with open('./Data/y_train_data_ref.pickle', 'rb') as f:
    y_train_data = pickle.load(f)

with open('./Data/x_val_data_ref.pickle', 'rb') as f:
    x_val_data = pickle.load(f)

with open('./Data/y_val_data_ref.pickle', 'rb') as f:
    y_val_data = pickle.load(f)

with open('./Data/dict_size_ref.pickle', 'rb') as f:
    size_dict = pickle.load(f)

## Manipulate validation data for batch size
batch_size = config.nbatch
dataset_train = TensorDataset(torch.from_numpy(x_train_data), torch.from_numpy(y_train_data))
dataset_val = TensorDataset(torch.from_numpy(x_val_data), torch.from_numpy(y_val_data))
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

# Set model
## Hyperparameter
emsize = size_dict
nhid = config.nhid
dropout = config.dropout
epochs = config.epochs
nlayers = config.nlayers
ntokens = size_dict
lr = config.lr
step_size = config.step_size # Period of learning rate decay
gamma = config.gamma # Multiplicative factor of learning rate decay
log_interval = 1000
best_val_loss = None
INPUT_DIM = size_dict
OUTPUT_DIM = size_dict
HIDDEN_DIM = 256 
ENC_LAYERS = config.enc_layers
DEC_LAYERS = config.dec_layers
ENC_HEADS = config.heads
DEC_HEADS = config.heads
ENC_PF_DIM = config.pf_layers 
DEC_PF_DIM = config.pf_layers 
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
PAD_IDX = 0

## Model SET
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(HIDDEN_DIM, size_dict)
model = Transformer(enc, dec, PAD_IDX, PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)
print(model)

## Functions for training

def evaluate(data_source):
    if cuda:
        model.cuda()

    model.eval()
    total_loss = 0.
    size = 0.
    start_time = time.time()
    with torch.no_grad():
        for i, samples in enumerate(val_loader):     
            x_val, y_val = samples
            data, targets = x_val.to(device), y_val.to(device) 

            output, _ = model(data)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)

            loss = criterion(output, torch.reshape(targets, (-1,)))

            total_loss += loss.item()
            size = i + 1

    return total_loss / size

def train(model, criterion, optimizer):
    if cuda:
        model.cuda()
    
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, samples in enumerate(train_loader):     
        x_train, y_train = samples
        data, targets = x_train.to(device), y_train.to(device)  

        optimizer.zero_grad()
        
        output, attention = model.forward(data)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        loss = criterion(output, torch.reshape(targets, (-1,)))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    
        optimizer.step()
    
        total_loss += loss.item()
    
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.7f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, (len(x_train_data)) // batch_size, optimizer.param_groups[0]["lr"],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
############################################

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=PAD_IDX)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

epochs_no_improve = 0
early_stop = False
try:
    model.zero_grad()                                          
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train(model=model, criterion=criterion, optimizer=optimizer)
        scheduler.step()
        val_loss = evaluate(val_loader)
        print('-' * 89) 
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        save = "./Transformer_encoder_ref.pth"
        save_2 = "./Transformer_encoder_ref_state.pth"
        if not best_val_loss or (val_loss + 1e-7) < best_val_loss:
            with open(save, 'wb') as f:
                torch.save(model, f)
            with open(save_2, 'wb') as e:
                torch.save(model.state_dict(), e)
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            if epoch > 30:
                epochs_no_improve += 1
        if epochs_no_improve == 5:
            print("Early stopping!")
            break
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


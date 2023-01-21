import torch

from tqdm.auto import tqdm
import pandas as pd
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch.nn.functional as F

def generate(src_field, trg_field, model, device, max_len=20, logging=True):
    model.eval()
    seq_len = 0
    attention_list = []
    with torch.no_grad():
        seq = ""
        src_list = []
        while(True):
            if seq_len == 100:
                break
            tokens = [src_field[SOS]] + src_list
            src_indexes = [src_field.index(token) for token in tokens]
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
            
            output, attention = model.forward(src_tensor)
            
            probs = [F.softmax(o, dim=-1) for o in output]
            
            # print("probs:", probs)
            ind_tops = [torch.multinomial(p, 1) for p in probs]
            
            for ch in ind_tops:      
                if ch[-1] == EOS:
                    return seq, attention, src_indexes
                else:
                    seq += trg_field[ch[-1]]
                    src_list.append(trg_field[ch[-1]])
                    seq_len += 1
                    break
    return seq, attention, src_indexes

def display_attention(src_field, translation, attention, src_indexes, idx, n_heads=8, n_rows=4, n_cols=2,):

    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    attention = attention.squeeze(0)[0].cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
    print(src_indexes)
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(['']  + [src_field[t] for t in src_indexes] , rotation=45)
    ax.axes.set_yticklabels([''] + [t for t in translation] + ['!'])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    
    plt.savefig('./Attention'+ str(idx) + '.png')

if __name__ == '__main__':
    PAD = 0
    SOS = 1
    EOS = 2

    cuda = False
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device) # change allocation of current GPU
        print ('Available devices ', torch.cuda.device_count())
        print ('Current cuda device ', torch.cuda.current_device())
        print(torch.cuda.get_device_name(device))
        print(torch.cuda.is_available())
        cuda = True
    else :
        print("NOT GPU")
        cuda = False

    with open('./Data/dict_size_ref.pickle', 'rb') as f:
        ntoken = pickle.load(f)
        print("ntoken", ntoken)
        print("open size_dict success")

    with open('./Data/dict_ref.pickle', 'rb') as f:
        vocab = pickle.load(f)
        print("vocab size", len(vocab))
        print("open vocab success")

    PATH = "Transformer_only_encoder.pth"

    # Loading
    model = torch.load(PATH)
    model.to("cuda")
    model.cuda()
    # print('latent_covid19 max_len = ', 97)
    # print(model)

    results = []
    for i in range(10000):
        translation, attention, src_indexes = generate(vocab, vocab, model, device, logging=True)    
        
        if translation != False:
            if i % 1000 == 0:
                print("{} : {}".format(i, translation))
            results.append(translation)
        
    samples = pd.DataFrame(results, columns=['SMILES'])
    samples.to_csv("Transformer_Encoder_FFNN_ref10.csv", index=False)
    print("done")

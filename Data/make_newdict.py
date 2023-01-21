import pickle

with open('./dict.pickle', 'rb') as f:
    vocab = pickle.load(f)
    print("open vocab success")

with open('./latent_dict.pickle', 'rb') as f:
    latent_vocab = pickle.load(f)

for i in range(len(vocab)):
    print(vocab[i], end=' ')

print()

for i in range(len(latent_vocab)):
    print(latent_vocab[i], end=' ')
print()

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

new_dict = Dictionary()

alist = []
for i in latent_vocab:
    if i not in vocab:
        print("vocab not containing :", end=' ')
        print(i)
        alist.append(i)

for i in range(len(vocab)):
    new_dict.add_word(vocab[i])
for i in range(len(alist)):
    new_dict.add_word(alist[i])

for i in range(len(new_dict)):
    print(new_dict.idx2char[i], end=' ')


with open('new_dict.pickle', 'wb') as f:
    pickle.dump(new_dict.idx2char, f)
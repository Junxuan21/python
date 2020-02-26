#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn, optim
import numpy as np
import re
import random
from collecitons import Counter


# In[ ]:


# load text data
with open(file_path) as f:
    text = f.read()


# In[ ]:


# data cleaning
def preprocess(text, freq):
    
    text = text.lower()
    text = text.replace('.', " ")
    #... more preprocessing
    
    # remove the low-freq words
    words = text.split()
    word_count = Counter(words)
    trimmed_words = [word for word in words if word_count[word] >= freq]
    
    return trimmed_words


# In[ ]:


words = preprocess(text, freq = 0)


# In[ ]:


# create a mapping table 
vocab = set(words)
vocab2int = {w: c for c,w in enumerate(list(vocab))}
int2vocab = {c: w for c,w in enumerate(list(vocab))}

int_words = [vocab2int[w] for w in words]
int_word_counts = Counter(int_words)
total_count = len(int_word_counts)

t = 1e-5
word_freq = {w: c/total_count for w, c in int_word_counts.items()}
prob_drop = {w: 1-np.sqrt(t/word_freq[w]) for w in int_word_counts)}
train_word = [w for w in int_words if random.random() < (1-prob_drop[w])]


# In[ ]:


# create a window to get a batch of words

def get_window(words, idx, window_size=5):
    # idx is the index of center word
    target_window = np.random.randint(1, window_zise+1)
    start_point = idx - target_window if idx - target_window > 0 else 0
    end_point = idx + target_window
    target = set(words[start_point:idx]+words[idx+1:end_point+1])
    
    return list(target)


# In[ ]:


# create batch iterator

def get_batch(words, batch_size, window_size=5):
    
    n_batches = len(words)//batch_size
    words = word[:n_batches*batch_size]
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [], []
        batch = words[idx:idx+batch_size]
        for i in range(len(batch)):
            x = batch[i]  # x is the center of each batch
            y = get_window(batch, i, window_size)  # y is the surrouding words of x
            batch_x.extend([x]*len(y))
            batch_y.extend(y)
            
        yield batch_x, batch_y


# In[ ]:


# build SkipGram language model using Negative Sampling

class SkipGramNeg(nn.Module):
    
    def __init__ (self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist # used in negative sampling 
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # initialize embedding tables with uniform distribution as this helps with covergence
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
    
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words) # input_words shape is like one hot
        return input_vectors 
    
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """
        generate noise vectors with shape(batch_size, n_samples, n_embed)
        """
        if self.noise_dist is None:
            # sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist, batch_size*n_samples, replacement=True)
        # resize noise_vectors to the matrix shape for operations
        noise_vectors = self.out_embed(noise_words).view(
                        batch_size, n_samples, self.n_embed) 
        
        return noise_vectors


# In[ ]:


# build loss function

class NegativeSamplingLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        # input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        # output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, 
                             input_vectors).sigmoid().log()
        # resulting shape: batch_size*1
        out_loss = out_loss.squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), 
                               input_vectors).sigmoid().log()
        # resulting shape: batch_size*n_samples*1
        # sum the loss over the sample of noise vectors
        noise_loss = noise_loss.squeeze().sum(1) 
        
        # negate and sum correct and noisy log-sigmoid loss
        # return average batch loss
        return -(out_loss + noise_loss).mean()


# In[ ]:


# uniform dist for negative sampling

word_freq = np.array(word_freq.values())
unigram_dist = word_freq/word_freq.sum()
noise_dist = torch.from_numpy(
    unigram_dist ** (0.75)/np.sum(unigram_dist ** (0.75))
)


# In[ ]:


# model instantiating
embedding_dim = 300
model = SkipGramNeg(len(vocab2int), embedding_dim, noise_dist=noise_dist)

# using the defined loss function
criterion = NegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 1500
steps = 0
epochs = 5
batch_size = 500
n_samples = 5


for e in range(epochs):
    
    # get input, target batches
    for input_words, target_words in get_batch(train_words, batch_size):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        
        # input, output and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(batch_size, n_samples)
        
        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        if steps //print_every == 0:
            print(loss)
            
        optimizer.zero_grad()  # reset gradient to zero
        loss.backward()
        optimizer.step()


# In[ ]:





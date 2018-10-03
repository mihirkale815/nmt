import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class ConcatAttention(nn.Module):
    def __init__(self,encoder_dim,decoder_dim):
        super(ConcatAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.linear = nn.Linear(self.encoder_dim+self.decoder_dim,self.decoder_dim)
        self.W = nn.Linear(self.decoder_dim,1)
        self.relu = nn.ReLU()

    def forward(self, hidden,encoder_outputs):
        batchsize,maxlen,encoderdim = encoder_outputs.size()
        #hidden = hidden.unsqueeze(1) #B,D => B,1,D
        hidden = hidden.expand(-1,maxlen,-1) #B,1,D => B,L,D
        input = torch.cat([encoder_outputs,hidden],dim=2)
        energy = self.relu(self.linear(input))
        scores = self.W(energy).squeeze(-1) #bs x maxlen
        scores = F.softmax(scores,dim=1)
        return scores



class RNNDecoder(nn.Module):

    def __init__(self,vocab,embed_size,context_size,hidden_size,n_layers,dropout,attention):
        super(RNNDecoder, self).__init__()
        self.dropout = dropout
        self.vocab = vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(len(self.vocab),self.embed_size)
        self.context_size = context_size
        self.rnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,
                               num_layers=self.n_layers, batch_first=True)
        self.embed_dropout = nn.Dropout(p=self.dropout)
        self.attention = attention
        self.linear = nn.Linear(self.hidden_size+self.context_size,len(vocab))

    def forward(self,input,hidden,encoder_inputs):
        embedding = self.embed(input)
        input = embedding
        output,hidden = self.rnn(input.unsqueeze(1),hidden)
        attn_scores = self.attention(hidden[0].permute(1, 0, 2), encoder_inputs).unsqueeze(1)  # bs x 1 x maxlen
        context = attn_scores.bmm(encoder_inputs).squeeze(1)
        output = torch.cat([output.squeeze(1),context],dim=1)
        output = F.log_softmax(self.linear(output),dim=1)
        return output,hidden,attn_scores.squeeze(1)


class RNNBahdanauDecoder(nn.Module):

    def __init__(self,vocab,embed_size,context_size,hidden_size,n_layers,dropout,attention):
        super(RNNBahdanauDecoder, self).__init__()
        self.dropout = dropout
        self.vocab = vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embed = nn.Embedding(len(self.vocab),self.embed_size)
        self.context_size = context_size
        self.rnn = nn.LSTM(input_size=self.embed_size+self.context_size,hidden_size=self.hidden_size,
                               num_layers=self.n_layers, batch_first=True)
        self.embed_dropout = nn.Dropout(p=self.dropout)
        self.attention = attention
        self.linear = nn.Linear(self.hidden_size+self.context_size,len(vocab))

    def forward(self,input,hidden,encoder_inputs):
        embedding = self.embed(input)
        attn_scores = self.attention(hidden[0].permute(1,0,2), encoder_inputs).unsqueeze(1)  # bs x 1 x maxlen
        context = attn_scores.bmm(encoder_inputs).squeeze(1)
        input = torch.cat([embedding,context],dim=1)
        output,hidden = self.rnn(input.unsqueeze(1),hidden)
        output = torch.cat([output.squeeze(1),context],dim=1)
        output = F.log_softmax(self.linear(output),dim=1)
        return output,hidden,attn_scores.squeeze(1)






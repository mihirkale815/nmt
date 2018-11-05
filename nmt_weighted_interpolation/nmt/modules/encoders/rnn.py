import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNEncoder(nn.Module):

    def __init__(self,vocab,embed_size,hidden_size,n_layers,bidirectional,dropout):
        super(RNNEncoder, self).__init__()
        self.dropout = dropout
        self.vocab = vocab
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(len(self.vocab),embed_size)
        self.rnn = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,bidirectional=self.bidirectional,num_layers=self.n_layers, batch_first=True)
        self.embed_dropout = nn.Dropout(p=self.dropout)
        self.encoder_dim = self.hidden_size*(2 if self.bidirectional else 1)
        self.tanh = nn.Tanh()
        self.projection = nn.Linear(self.encoder_dim,self.hidden_size)


    def forward(self, batch,lens):
        embeddings = self.embed(batch)
        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output,_ = pad_packed_sequence(packed_output, batch_first=True)
        if self.bidirectional :
            c = torch.cat([c for c in hidden[1]],dim=1)
            c = self.tanh(self.projection(c)).unsqueeze(0)
            h = hidden[0][0].unsqueeze(0)
            hidden = (c,h)
        return output,hidden

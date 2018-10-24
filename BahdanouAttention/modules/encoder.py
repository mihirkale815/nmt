import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, n_layers=1, dropout=0.):
        super(RNNEncoder, self).__init__()
        self.num_layers = n_layers
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size)
        self.rnn = nn.GRU(self.embed_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout)
        self.projection = nn.Linear(2*hidden_size, hidden_size, bias=True)

    def forward(self, batch, lens):
        embeddings = self.embedding(batch)
        packed_input = pack_padded_sequence(embeddings, lens, batch_first=True)
        packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        forward_hid = hidden[0:hidden.size(0):2]
        backward_hid = hidden[1:hidden.size(0):2]

        final_hid = torch.cat([forward_hid, backward_hid], dim=2)
        #encoder_hid = torch.tanh(self.projection(final_hid))

        return output, final_hid

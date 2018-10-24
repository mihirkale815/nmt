import torch.nn as nn
import torch


class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, attention, vocab, num_layers=1, dropout=0.):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention = attention
        self.vocab = vocab

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size)
        self.rnn = nn.GRU(self.embed_size+2*self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.pre_out_layer = nn.Linear(self.hidden_size+2*self.hidden_size+self.embed_size, self.hidden_size, bias=False)
        self.linear = nn.Linear(self.hidden_size, len(vocab), bias=False)
        self.projection = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True)

    def forward(self, target, max_len, src_encodings, encoder_hid, source_mask, target_mask):
        target_embedding = self.embedding(target)
        key = self.attention.key_linear(src_encodings)
        hidden = self.init_hidden(encoder_hid)
        decoder_outputs = []
        decoder_pre_outputs = []

        for i in range(max_len):
            prev_word_embed = target_embedding[:, i].unsqueeze(1)
            query = hidden[-1].unsqueeze(1)
            context, attn_scores = self.attention(query, key, src_encodings, source_mask)
            rnn_input = torch.cat([prev_word_embed, context], dim=2)
            output, hidden = self.rnn(rnn_input, hidden)
            pre_output = torch.cat([prev_word_embed, output, context], dim=2)
            #pre_output = self.dropout_layer(pre_output)
            pre_output = self.pre_out_layer(pre_output)
            decoder_outputs.append(output)
            decoder_pre_outputs.append(pre_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_pre_outputs = torch.cat(decoder_pre_outputs, dim=1)
        #decoder_pre_outputs = self.linear(decoder_pre_outputs)


        return decoder_outputs, hidden, decoder_pre_outputs, target

    def init_hidden(self, encoder_final):
        return torch.tanh(self.projection(encoder_final))




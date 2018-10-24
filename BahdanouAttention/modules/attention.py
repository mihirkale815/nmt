import torch.nn as nn
import torch
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, key_size, query_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.key_size = 2*key_size
        self.query_size = query_size

        self.key_linear = nn.Linear(self.key_size, hidden_size, bias=False)
        self.query_linear = nn.Linear(self.query_size, hidden_size, bias=False)
        self.energy_linear = nn.Linear(self.hidden_size, 1, bias=False)

        self.attn_scores = None

    def forward(self, query, key, value, mask):
        query = self.query_linear(query)
        scores = self.energy_linear(torch.tanh(query+key))
        scores = scores.squeeze(2).unsqueeze(1)

        scores.data.masked_fill_(mask == 0, -float('inf'))

        attn_scores = F.softmax(scores, dim=-1)
        self.attn_scores = attn_scores
        context = torch.bmm(attn_scores, value)

        return context, self.attn_scores
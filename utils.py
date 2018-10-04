import math
from typing import List

import numpy as np
import torch

def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    data = sorted(data,key = lambda x : len(x[1]) )
    datalen = len(data)

    for i in range(batch_num):
        #indices = index_array[i * batch_size: (i + 1) * batch_size]
        indices = [idx for idx in range(i * batch_size,min(datalen,(i + 1) * batch_size))]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]


        #for src_sent,tgt_sent in zip(src_sents,tgt_sents):
            #print(src_sent,tgt_sent)

        yield src_sents, tgt_sents


def convert_to_tensor(src_sents,src_vocab,tgt_sents,tgt_vocab):

    lens = np.array([len(sent) for sent in src_sents])
    sorted_indices = np.argsort(-lens)

    def F(sents,vocab,indices):
        raw = [sents[i] for i in indices]
        lens = np.array([len(sent) for sent in sents])
        sents = [ torch.LongTensor([vocab[word] for word in sent]) for sent in sents]
        sents = torch.nn.utils.rnn.pad_sequence(sents,batch_first=True,padding_value=vocab['<pad>'])
        lens = lens[indices]
        sents = sents[indices]
        return raw,sents,lens

    src_sents,src_tensor, src_lens = F(src_sents,src_vocab,sorted_indices)
    tgt_sents,tgt_tensor, tgt_lens = F(tgt_sents,tgt_vocab,sorted_indices)

    return src_tensor,src_lens,tgt_tensor,tgt_lens


def convert_to_tensor_single(sent,vocab):
    sent = torch.LongTensor([vocab[word] for word in sent])
    return sent,len(sent)
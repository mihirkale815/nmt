#import spacy
#from torchtext import data, datasets

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import read_corpus, batch_iter
import utils
import pickle
import os
from vocab import Vocab, VocabEntry
from tqdm import tqdm
import sys
from modules.encoder import RNNEncoder
from modules.decoder import RNNDecoder
from modules.attention import BahdanauAttention

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
cuda = True

device = torch.device("cuda" if cuda else "cpu")
DEVICE=torch.device('cuda:0')


def isFloat(x):
    try:
        float(x)
        return True
    except:
        return False


def create_embed_matrix(file_path, vocab):

    embedding_dim = 300
    vocab_size = len(vocab)
    print(vocab_size, 'vocab size')
    embeddings_index = {}
    f = open(file_path)
    i = 0

    for line in f:
        values = line.split()
        if len(values) < 3:
            continue
        i += 1
        num_words = 0
        for j in range(len(values)):
            if isFloat(values[j]):
                break
            else:
                num_words += 1

        coefs = np.asarray(values[num_words:], dtype='float32')
        for idx in range(num_words):
            embeddings_index[values[idx]] = coefs
    print('done reading embedding file')

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word in vocab.word2id:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[vocab.word2id[word]] = embedding_vector
    print("returning")
    return embedding_matrix


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_vocab, tgt_vocab, embed_size, generator, use_pretrained=False, embed_matrix_src=None, embed_matrix_tgt=None):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab_size = src_vocab
        self.tgt_vocab_size = tgt_vocab
        self.generator = generator
        self.embed_size = embed_size

        self.encoder_embed = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.decoder_embed = nn.Embedding(self.tgt_vocab_size, self.embed_size)
        if use_pretrained:
            self.encoder_embed.weight.data.copy_(torch.Tensor(embed_matrix_src))
            self.decoder_embed.weight.data.copy_(torch.Tensor(embed_matrix_tgt))

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        #print(encoder_hidden.size(), encoder_final.size(), trg.size())
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        print(type(self.encoder))
        return self.encoder(self.encoder_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.decoder_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=num_layers, batch_first=True)
        #self.rnn = nn.GRU(input_size, hidden_size, num_layers,
         #                 batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional GRU to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        #print('final',final.size())
        # we need to manually concatenate the final states for both directions
        fwd_final = final[0][0:final[0].size(0):2]
        bwd_final = final[0][1:final[0].size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.rnn = nn.LSTM(input_size=emb_size + 2 * hidden_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        #self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
         #                 batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""
        #print('hidden', hidden.size())
        # compute context vector using attention mechanism
        query = hidden[0][-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        #print(query.size())
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, trg_embed, encoder_hidden, encoder_final,
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        #if hidden is None:
         #   hidden = self.init_hidden(encoder_final)
        if hidden is None:
            hidden, cell_state = self.init_hidden(encoder_final)
            decoder_hidden = (hidden, cell_state)
        else:
            decoder_hidden = hidden
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, decoder_hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, decoder_hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, decoder_hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        #print('encoder_final', encoder_final.size())
        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final)), self.bridge(encoder_final)


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


def make_model(src_vocab, tgt_vocab, embed_matrix_src, embed_matrix_tgt, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        RNNEncoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        RNNDecoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        src_vocab, tgt_vocab, emb_size,
        Generator(hidden_size, tgt_vocab), use_pretrained=False, embed_matrix_src=embed_matrix_src, embed_matrix_tgt=embed_matrix_tgt)

    return model.to(device)


class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        self.trg = None
        self.trg_y = None
        self.trg_mask = None
        self.trg_lengths = None
        self.ntokens = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trzzzg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if cuda:
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()


def run_epoch(train_data, vocab, model, loss_compute, print_every=10):
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    i = 0

    for src_sents, tgt_sents in batch_iter(train_data, batch_size=64, shuffle=True):
        i += 1
        src_sents, src_lens, tgt_sents, tgt_lens, source_mask, target_mask = utils.convert_to_tensor(src_sents, vocab.src, tgt_sents, vocab.tgt)
        #print(src_sents.size(), tgt_sents.size())
        tgt_y = tgt_sents[:, 1:]
        tgt = tgt_sents[:, :-1]
        out, _, pre_output = model.forward(src_sents, tgt, source_mask, target_mask, src_lens, tgt_lens)
        #print(pre_output.size(), tgt_y.size())

        #print(pre_output)
        #print(tgt_y)
        loss = loss_compute(pre_output, tgt_y, 64)
        print(loss)
        total_loss += loss
        ntokens = (tgt_sents!= 0).data.sum().item()

        total_tokens += ntokens
        print_tokens += ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / 64, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0
            print('perplexity: ', math.exp(total_loss/total_tokens))


    return math.exp(total_loss / float(total_tokens))


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

        return loss.data.item() * norm


def greedy_decode(model, src, src_mask, src_lengths, max_len=70, sos_index=1, eos_index=None):
    """Greedily decode a sentence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None
    #print(encoder_hidden.size(), encoder_final.size(), prev_y.size())
    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())

    output = np.array(output)

    # cut off everything starting from </s>
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output, np.concatenate(attention_scores, axis=1)


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.id2word[int(i)] for i in x]

    return [str(t) for t in x]


def print_examples(valid_data, model, vocab, n=2, max_len=70,
                   sos_index=1,
                   src_eos_index=None,
                   trg_eos_index=None,
                   src_vocab=None, trg_vocab=None):
    """Prints N examples. Assumes batch size of 1."""

    model.eval()
    count = 0
    print()

    i = 0
    for src_sents, tgt_sents in batch_iter(valid_data, batch_size=1, shuffle=False):
        i += 1
        src_sents, src_lens, tgt_sents, tgt_lens, source_mask, target_mask = utils.convert_to_tensor(src_sents, vocab.src, tgt_sents, vocab.tgt)
        src = src_sents.cpu().numpy()[0, :]
        trg = tgt_sents.cpu().numpy()[0, :]

        src = src[:-1] if src[-1] == vocab.src['</s>'] else src
        trg = trg[:-1] if trg[-1] == vocab.tgt['</s>'] else trg

        result, _ = greedy_decode(
            model, src_sents, source_mask, src_lens,
            max_len=max_len, sos_index=vocab.tgt['<s>'], eos_index=vocab.tgt['</s>'])

        print("Example #%d" % (i + 1))
        print("Src : ", " ".join(lookup_words(src, vocab=vocab.src)))
        print("Trg : ", " ".join(lookup_words(trg, vocab=vocab.tgt)))
        print("Pred: ", " ".join(lookup_words(result, vocab=vocab.tgt)))
        print()

        count += 1
        if count == n:
            break


def write_preds(src_data, model, vocab, file_name, max_len):
    model.eval()
    f = open(file_name, 'w')
    #f2 = open('true.txt', 'w')
    for src_sent in tqdm(src_data, desc='Decoding', file=sys.stdout):
        src_sent, src_len, source_mask = utils.convert_to_tensor_single(src_sent, vocab.src)
        src = src_sent.cpu().numpy()[0, :]
        #trg = tgt_sent.cpu().numpy()[0, :]
        result, _ = greedy_decode(
            model, src_sent, source_mask, src_len,
            max_len=max_len, sos_index=vocab.tgt['<s>'], eos_index=vocab.tgt['</s>'])
        f.write(" ".join(lookup_words(result, vocab=vocab.tgt))+'\n')
        #f2.write(" ".join(lookup_words(trg, vocab=vocab.tgt))+'\n')

    f.close()
    #f2.close()
        
    #for src_sents, tgt_sents in batch_iter(data, batch_size=1, shuffle=False):
        
     #   src_sents, src_lens, tgt_sents, tgt_lens, source_mask, target_mask = utils.convert_to_tensor(src_sents, vocab.src, tgt_sents, vocab.tgt)
     #   src = src_sents.cpu().numpy()[0, :]
     #   trg = tgt_sents.cpu().numpy()[0, :]

     #   src = src[:-1] if src[-1] == vocab.src['</s>'] else src
     #   trg = trg[:-1] if trg[-1] == vocab.tgt['</s>'] else trg

     #   result, _ = greedy_decode(
      #      model, src_sents, source_mask, src_lens,
       #     max_len=max_len, sos_index=vocab.tgt['<s>'], eos_index=vocab.tgt['</s>'])
     #   f.write(" ".join(lookup_words(result, vocab=vocab.tgt))+'\n')
     #   f2.write(" ".join(lookup_words(trg, vocab=vocab.tgt))+'\n')

  #  f.close()
  #  f2.close()

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, pad_idx)


def train(model, vocab, train_data, valid_data, test_data, test_data_src, model_save_path, num_epochs=10, lr=0.001, print_every=10):
    """Train a model on IWSLT"""
    print('starting training')
    if cuda:
        model.cuda()
    patience_thresh = 1
    max_num_trial = 5
    # optionally add label smoothing; see the Annotated Transformer
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)
    #optim = torch.optim.Adam(model.parameters(), lr=lr)
    hist_valid_scores = []
    dev_perplexities = []
    epoch = 0
    num_trial = 0
    while True:
    #for epoch in range(num_epochs):
        epoch += 1
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch(train_data, vocab, model, SimpleLossCompute(model.generator, criterion, optimizer), print_every=print_every)

        model.eval()
        with torch.no_grad():
            print_examples(valid_data, model, vocab)
            # print_examples((rebatch(PAD_INDEX, x) for x in valid_iter),
            #                model, n=3, src_vocab=SRC.vocab, trg_vocab=TRG.vocab)
            dev_perplexity = run_epoch(valid_data, vocab, model, SimpleLossCompute(model.generator, criterion, None), print_every=print_every)
            # dev_perplexity = run_epoch((rebatch(PAD_INDEX, b) for b in valid_iter),
            #                            model,
            #                            SimpleLossCompute(model.generator, criterion, None))
            valid_metric = -dev_perplexity
            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)
            if is_better:
                patience = 0
                print('save currently the best model to [%s]' % model_save_path)#, file=sys.stderr)
                    #model.save(model_save_path)
                torch.save(model.state_dict(), model_save_path)
            elif patience < patience_thresh:
                patience += 1
                if patience == patience_thresh:
                    num_trial += 1
                    #if num_trial == max_num_trial:
                     #   print('early stop')
                      #  write_preds(test_data_src, model, vocab, 'decode.txt', 70)
                       # exit(0)
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("lr for param group =", param_group['lr'])
                    model.load_state_dict(torch.load(model_save_path))
                    patience = 0
        if epoch == num_epochs:
            print('reached maximum number of epochs!')#, file=sys.stderr)
            write_preds(test_data_src, model, vocab, 'new_decode.txt', 70)
            exit(0)
        print("Validation perplexity: %f" % dev_perplexity)
        dev_perplexities.append(dev_perplexity)
    write_preds(test_data_src, model, vocab, 'new_decode.txt', 70)
    torch.save(model.state_dict(), model_save_path)
    #return train_perplexity
    return dev_perplexities


def write_predictions(model, test_data, vocab, file_name):
    write_preds(test_data, model, vocab, file_name)

def main():

    #model_path = 'model_state_dict'

    train_data_src = read_corpus('data/train.en-gl.gl.txt', source='src')
    train_data_tgt = read_corpus('data/train.en-gl.en.txt', source='tgt')

    dev_data_src = read_corpus('data/dev.en-gl.gl.txt', source='src')
    dev_data_tgt = read_corpus('data/dev.en-gl.en.txt', source='tgt')

    test_data_src = read_corpus('data/test.en-gl.gl.txt', source='src')
    test_data_tgt = read_corpus('data/test.en-gl.en.txt', source='tgt')

    num_samples = len(test_data_src)
    test_data_src = test_data_src[0:num_samples]
    test_data_tgt = test_data_tgt[0:num_samples]

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    test_data = list(zip(test_data_src, test_data_tgt))

    train_batch_size = 64
    clip_grad = float(5)
    valid_niter = 180
    log_every = 10
    model_save_path = os.path.join('experiments/', 'model_state_dict')
    #model_save_path = os.path.join('experiments/saved_models/', 'model_state_dict')

#    f = open('data/vocab.bin', 'rb')
 #   print('file opened')
  #  vocab = pickle.load(f)
   # print('vocab loaded')

    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    print('vocab loaded')
#    embed_matrix_src = create_embed_matrix('data/wiki.gl.vec', vocab.src)
    print('source embed matrix created')

 #   embed_matrix_tgt = create_embed_matrix('data/wiki.en.vec', vocab.tgt)
    print('target embed matrix created')
    embed_matrix_src = None
    embed_matrix_tgt = None
    model = make_model(len(vocab.src), len(vocab.tgt), embed_matrix_src, embed_matrix_tgt,
                       emb_size=300, hidden_size=256,
                       num_layers=1, dropout=0.2)
#    model.load_state_dict(torch.load(model_save_path))    
    print('created model')
    dev_perplexities = train(model, vocab, train_data, dev_data, test_data, test_data_src, model_save_path, print_every=10)
    write_preds(test_data_src, model, vocab, 'mono_new.txt', 70)

main()

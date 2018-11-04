
"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> --test-src=<file> --test-tgt=<file> --beam_size=<int> --model_path=<file> --out-file=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] --vocab=<file> MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --test-src=<file>
    --test-tgt=<file>
    --beam_size=<int>
    --model_path=<file>
    --out-file=<file>
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]


"""

from docopt import docopt
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
args = docopt(__doc__)
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


class NMT(nn.Module):

    def __init__(self, vocab, embed_size, hidden_size, generator, use_pretrained=False, embed_matrix_src=None, embed_matrix_tgt=None):
        super(NMT, self).__init__()
        
        self.src_vocab_size = len(vocab.src)
        self.tgt_vocab_size = len(vocab.tgt)
        self.generator = generator
        self.embed_size = embed_size
        self.attention = BahdanauAttention(hidden_size)
        self.encoder = RNNEncoder(embed_size, hidden_size, num_layers=1, dropout=0.2)
        self.decoder = RNNDecoder(embed_size, hidden_size, self.attention, num_layers=1, dropout=0.2)
        self.encoder_embed = nn.Embedding(self.src_vocab_size, self.embed_size)
        self.decoder_embed = nn.Embedding(self.tgt_vocab_size, self.embed_size)
        if use_pretrained:
            self.encoder_embed.weight.data.copy_(torch.Tensor(embed_matrix_src))
            self.decoder_embed.weight.data.copy_(torch.Tensor(embed_matrix_tgt)) 

        self.criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.proj = nn.Linear(256, len(vocab.tgt), bias=False)

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.encoder_embed(src), src_mask, src_lengths)

    def decode(self, encoder_hidden, encoder_final, src_mask, trg, trg_mask,
               decoder_hidden=None):
        return self.decoder(self.decoder_embed(trg), encoder_hidden, encoder_final,
                            src_mask, trg_mask, hidden=decoder_hidden)

    def loss(self, x, y, norm, criterion, optimizer):
        x = F.log_softmax(self.proj(x), dim=-1)
        loss = criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.data.item() * norm


class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def run_epoch(train_data, vocab, model, loss_compute, print_every=10):

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    i = 0

    for src_sents, tgt_sents in batch_iter(train_data, batch_size=64, shuffle=True):
        i += 1
        src_sents, src_lens, tgt_sents, tgt_lens, source_mask, target_mask = utils.convert_to_tensor(src_sents, vocab.src, tgt_sents, vocab.tgt)
        tgt_y = tgt_sents[:, 1:]
        tgt = tgt_sents[:, :-1]
        out, _, pre_output = model.forward(src_sents, tgt, source_mask, target_mask, src_lens, tgt_lens)
        loss = loss_compute(pre_output, tgt_y, 64)
        total_loss += loss
        ntokens = (tgt_sents!= 0).data.sum().item()

        total_tokens += ntokens
        print_tokens += ntokens

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print('epoch %d, avg. loss %.4f, avg. ppl %.4f, speed %.2f words/sec, time elapsed %.2f sec' % (i, loss / 64, math.exp(total_loss/total_tokens), (print_tokens / elapsed), elapsed))
            #elapsed = time.time() - start
            #print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
             #     (i, loss / 64, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0
            #print('perplexity: ', math.exp(total_loss/total_tokens))


    return math.exp(total_loss / float(total_tokens))


class SimpleLossCompute:

    def __init__(self, generator, criterion, opt=None, vocab=None):
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

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_lengths)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    hidden = None
    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask,
                prev_y, trg_mask, hidden)

            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)

    output = np.array(output)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]

    return output


def lookup_words(x, vocab=None):
    if vocab is not None:
        x = [vocab.id2word[int(i)] for i in x]

    return [str(t) for t in x]


def print_examples(valid_data, model, vocab, n=2, max_len=70,
                   sos_index=1,
                   src_eos_index=None,
                   trg_eos_index=None,
                   src_vocab=None, trg_vocab=None):

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

        result = greedy_decode(
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
    for src_sent in tqdm(src_data, desc='Decoding', file=sys.stdout):
        src_sent, src_len, source_mask = utils.convert_to_tensor_single(src_sent, vocab.src)
        src = src_sent.cpu().numpy()[0, :]
        result = greedy_decode(
            model, src_sent, source_mask, src_len,
            max_len=max_len, sos_index=vocab.tgt['<s>'], eos_index=vocab.tgt['</s>'])
        f.write(" ".join(lookup_words(result, vocab=vocab.tgt))+'\n')

    f.close()


def train(model, vocab, train_data, valid_data, test_data, test_data_src, model_save_path, num_epochs=30, lr=0.001, print_every=10):
    print('starting training')
    if cuda:
        model.cuda()
    patience_thresh = 1
    max_num_trial = 5
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)
    hist_valid_scores = []
    dev_perplexities = []
    epoch = 0
    num_trial = 0
    while True:
        epoch += 1
        print("Epoch", epoch)
        model.train()
        train_perplexity = run_epoch(train_data, vocab, model, SimpleLossCompute(model.generator, criterion, optimizer, vocab), print_every=print_every)

        model.eval()
        with torch.no_grad():
            print_examples(valid_data, model, vocab)
            dev_perplexity = run_epoch(valid_data, vocab, model, SimpleLossCompute(model.generator, criterion, None, vocab), print_every=print_every)
            valid_metric = -dev_perplexity
            is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
            hist_valid_scores.append(valid_metric)
            if is_better:
                patience = 0
                print('save currently the best model to [%s]' % model_save_path)
                torch.save(model.state_dict(), model_save_path)
            elif patience < patience_thresh:
                patience += 1
                if patience == patience_thresh:
                    num_trial += 1
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("lr for param group =", param_group['lr'])
                    model.load_state_dict(torch.load(model_save_path))
                    patience = 0
        if epoch == num_epochs:
            print('reached maximum number of epochs!')
            write_preds(test_data_src, model, vocab, 'decode_gl_concat_embed.txt', 70)
            exit(0)
        print("Validation perplexity: %f" % dev_perplexity)
        dev_perplexities.append(dev_perplexity)
    write_preds(test_data_src, model, vocab, 'decode_gl_concat_embed.txt', 70)
    torch.save(model.state_dict(), model_save_path)
    
    return dev_perplexities


def main():


    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    test_data_src = read_corpus(args['--test-src'], source='src')
    test_data_tgt = read_corpus(args['--test-tgt'], source='tgt')

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
    model_save_path = os.path.join('experiments/', 'model_gl_concat_embed')
    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    embed_matrix_src = create_embed_matrix('data/wiki.gl.vec', vocab.src)
    print('source embed matrix created')

    embed_matrix_tgt = create_embed_matrix('data/wiki.en.vec', vocab.tgt)
    print('target embed matrix created')
    #embed_matrix_src = None
    #embed_matrix_tgt = None
    generator = Generator(256, len(vocab.tgt))
  #  model = NMT(vocab=vocab, embed_size=300, hidden_size=256, generator=generator)
    model = NMT(vocab=vocab, embed_size=300, hidden_size=256, generator=generator, use_pretrained=True, embed_matrix_src=embed_matrix_src, embed_matrix_tgt=embed_matrix_tgt)
#    model.load_state_dict(torch.load(model_save_path))    
    print('created model')
    dev_perplexities = train(model, vocab, train_data, dev_data, test_data, test_data_src, model_save_path, print_every=10)
#    write_preds(test_data_src, model, vocab, 'mono_new.txt', 70)

main()

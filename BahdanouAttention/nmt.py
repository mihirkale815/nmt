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

import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union, Any
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry
import torch.nn as nn
import torch
from modules.encoder import RNNEncoder
from modules.decoder import RNNDecoder
from modules.attention import BahdanauAttention
import utils
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
# import torch.cuda as cuda
# from beam import Beam
# from torch.autograd import Variable
# import pdb
#
# import beam as Beam_Class
import torch.nn.functional as F

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
args = docopt(__doc__)
device = torch.device("cuda" if args['--cuda'] else "cpu")


class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return self.proj(x)


class NMT(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout, encoder_layers, decoder_layers, generator):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.vocab = vocab
        self.encoder_num_layers = encoder_layers
        self.decoder_num_layers = decoder_layers
        self.generator = generator

        self.loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.vocab.tgt.word2id['<pad>']).to(device)

        self.attention = BahdanauAttention(self.hidden_size, self.hidden_size, self.hidden_size)
        self.encoder = RNNEncoder(self.embed_size, self.hidden_size, self.vocab.src, self.encoder_num_layers, self.dropout)
        self.decoder = RNNDecoder(self.embed_size, self.hidden_size, self.attention, self.vocab.tgt, self.decoder_num_layers)

    def forward(self, src_sents, tgt_sents):
        src_sents, src_lens, tgt_sents, tgt_lens, source_mask, target_mask = utils.convert_to_tensor(src_sents, self.vocab.src, tgt_sents, self.vocab.tgt)
        source_mask = source_mask.to(device)
        target_mask = target_mask.to(device)
        src_encodings, encoder_final = self.encode(src_sents, src_lens)
        tgt = tgt_sents[:, :-1]
        tgt_y = tgt_sents[:, 1:]
        return self.decode(src_encodings, encoder_final, source_mask, target_mask, tgt), tgt_y

    def encode(self, source, src_lens):
        source = source.to(device)
        src_encodings, encoder_final = self.encoder(source, src_lens)
        return src_encodings, encoder_final

    def decode(self, src_encodings, encoder_final, source_mask, target_mask, target):
        target = target.to(device)
        max_len = target.size()[1]
        return self.decoder(target, max_len, src_encodings, encoder_final, source_mask, target_mask)

    def greedy_decode(self, src_sent, src_len, max_len, src_mask):
        bos, eos = self.vocab.tgt.word2id['<s>'], self.vocab.tgt['</s>']
        with torch.no_grad():
            src_encodings, decoder_init_state = self.encode(src_sent, src_len)
            prev_y = torch.ones(1, 1).fill_(bos).type_as(src_sent)
            trg_mask = torch.ones_like(prev_y)

        output = []

        for i in range(max_len):
            with torch.no_grad():
                out, hidden, pre_output, target = self.decode(src_encodings, decoder_init_state, src_mask, trg_mask, prev_y)

                prob = self.generator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data.item()
            output.append(next_word)
            prev_y = torch.ones(1, 1).type_as(src_sent).fill_(next_word)

        output = np.array(output)

        first_eos = np.where(output == eos)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]


        return output

    def lookup_words(self, x, vocab):
        if vocab is not None:
            x = [vocab.id2word[int(i)] for i in x]

        return [str(t) for t in x]

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int = 32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size

        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            (out, _, predictions, targets), tgt_y = self.forward(src_sents, tgt_sents)
            predictions = self.generator(predictions)
            batch_size, max_len, vocab_size = predictions.size()
            predictions = predictions.contiguous().view(batch_size * max_len, vocab_size)
            targets = tgt_Y.contiguous().view(batch_size * max_len)

            loss = self.loss(predictions, targets)
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str, generator):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        vocab = pickle.load(open("data/vocab.bin", 'rb'))
        model = NMT(embed_size=int(args['--embed-size']), hidden_size=int(args['--hidden-size']), vocab=vocab, dropout=float(args['--dropout']), encoder_layers=1, decoder_layers=1,
                    generator=generator).to(device)
        model.load_state_dict(torch.load(model_path))
        return model

    def greedy_search(self, src_sent: List[str], max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """
        Given a single source sentence, perform greedy search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sent, src_len, source_mask = \
            utils.convert_to_tensor_single(src_sent, self.vocab.src)

        hyps = self.greedy_decode(src_sent, src_len, max_decoding_time_step, source_mask)
        return hyps


def train(args):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    print(args['--test-src'], args['--train-src'])
    print(type(args['--test-src']))
    print(type(args['--train-src']))
    test_data_src = read_corpus(args['--test-src'], source='src')
    test_data_tgt = read_corpus(args['--test-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    test_data = list(zip(test_data_src, test_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = os.path.join(args['--save-to'], 'model_state_dict')

    vocab = pickle.load(open(args['--vocab'], 'rb'))
    generator = Generator(hidden_size=int(args['--hidden-size']), vocab_size=len(vocab.tgt))
    model = NMT(embed_size=int(args['--embed-size']), hidden_size=int(args['--hidden-size']), vocab=vocab, dropout=float(args['--dropout']), encoder_layers=1, decoder_layers=1, generator=generator)
    model.to(device)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Starting Maximum Likelihood training')

    lr = float(args['--lr'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5, last_epoch=-1)


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            torch.set_grad_enabled(True)
            model.train()

            train_iter += 1
            (out, _, predictions, targets), tgt_y = model(src_sents, tgt_sents)
            predictions = generator(predictions)
            batch_size, max_len, vocab_size = predictions.size()

            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            targets = tgt_y.contiguous().view(-1)
            loss = model.loss(predictions, targets)
            report_loss += loss.item()
            cum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.4f, avg. ppl %.4f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time))  # , file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples))#, file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')#, file=sys.stderr)

                # compute dev. ppl and bleu
                torch.no_grad()
                model.eval()
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))#, file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)


                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path)#, file=sys.stderr)
                    #model.save(model_save_path)
                    torch.save(model.state_dict(), model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience)#, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial)#, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!',)# file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        scheduler.step()
                        print("Modified learning rate...")
                        print('loading previously best model and decaying learning rate')#, file=sys.stderr)
                        for param_group in optimizer.param_groups:
                            print("lr for param group =", param_group['lr'])


                        # load model
                        model.load_state_dict(torch.load(model_save_path))

                        #print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!')#, file=sys.stderr)
                    exit(0)

            # if train_iter % valid_niter == 0:
            #     print("running decoding on text")
            #     decode_epoch(args, epoch)
            #     print("calculating perplexity for test")
            #     cum_loss = cumulative_examples = cumulative_tgt_words = 0.
            #     torch.no_grad()
            #     model.eval()
            #     test_ppl = model.evaluate_ppl(test_data, batch_size=128)
            #     valid_metric = -test_ppl
            #     print('test: iter %d, test. ppl %f' % (train_iter, test_ppl))#, file=sys.stderr)


def greedy_search(model: NMT, test_data_src: List[List[str]], max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.greedy_search(src_sent, max_decoding_time_step=max_decoding_time_step)
        hypotheses.append(example_hyps)
    return hypotheses


def decode_epoch(args, epoch):
    test_data_src = read_corpus(args['--test-src'], source='src')
    if args['--test-tgt']:
        test_data_tgt = read_corpus(args['--test-tgt'], source='tgt')
    num_samples = len(test_data_src)  # 10
    test_data_src = test_data_src[0:num_samples]
    test_data_tgt = test_data_tgt[0:num_samples]

    print("load model from {args['--model_path']}", file=sys.stderr)
    model = NMT.load(args['--model_path'])

    hypotheses = greedy_search(model, test_data_src,
                               max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['--test-tgt']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU:', bleu_score, file=sys.stderr)

    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    output_file = args['--out-file'] + '/decode_' + str(epoch) + '.txt'
    with open(output_file, 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0].value
            top_hyp = [vocab.tgt.id2word[int(word)] for word in top_hyp]
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    #    pdb.set_trace()
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    num_samples = len(test_data_src)  # 10
    test_data_src = test_data_src[0:num_samples]
    test_data_tgt = test_data_tgt[0:num_samples]
    vocab = pickle.load(open(args['--vocab'], 'rb'))
    generator = Generator(hidden_size=int(args['--hidden-size']), vocab_size=len(vocab.tgt))
    print("load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'], generator)

    # hypotheses = beam_search(model, test_data_src,
    #                        beam_size=int(args['--beam-size']),
    #                       max_decoding_time_step=int(args['--max-decoding-time-step']))

    hypotheses = greedy_search(model, test_data_src,
                               max_decoding_time_step=int(args['--max-decoding-time-step']))


    # if args['TEST_TARGET_FILE']:
    #     top_hypotheses = [hyps[0] for hyps in hypotheses]
    #     bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    #     print('Corpus BLEU:', bleu_score, file=sys.stderr)

    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    with open(args['OUTPUT_FILE'], 'w') as f:
        for hyp in hypotheses:
            hyp_sent = ' '.join(model.lookup_words(hyp, vocab.tgt))
            f.write(hyp_sent + '\n')
        # for src_sent, hyps in zip(test_data_src, hypotheses):
        #     top_hyp = hyps[0].value
        #     top_hyp = [vocab.tgt.id2word[int(word)] for word in top_hyp]
        #     hyp_sent = ' '.join(top_hyp)
        #     f.write(hyp_sent + '\n')


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def main():
    if args['train']:
        train(args)

    elif args['decode']:
        decode(args)


if __name__ == '__main__':
    main()




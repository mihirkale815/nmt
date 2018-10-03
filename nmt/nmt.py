    # coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
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
from modules.encoders.rnn import RNNEncoder
from modules.decoders.rnn import RNNDecoder,ConcatAttention
import utils
from torch import Tensor
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import torch.cuda as cuda
from beam import Beam
from torch.autograd import Variable
import pdb

import beam as Beam_Class

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
args = docopt(__doc__)
device = torch.device("cuda" if args['--cuda'] else "cpu")

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.n_enc_layers = 1
        self.n_dec_layers = 1
        self.bidirectional = False
        self.attn_context_size = (2 if self.bidirectional else 1) * self.hidden_size
        # initialize neural network layers...
        self.loss = nn.NLLLoss(ignore_index=self.vocab.tgt.word2id['<pad>']).to(device)
        self.attention = ConcatAttention(encoder_dim=self.hidden_size,decoder_dim=self.hidden_size)

        self.encoder = RNNEncoder(vocab=self.vocab.src,embed_size=self.embed_size,bidirectional=self.bidirectional,
                                  hidden_size=self.hidden_size,n_layers=self.n_enc_layers,dropout=self.dropout_rate)
        self.decoder = RNNDecoder(vocab=self.vocab.tgt,embed_size=self.embed_size,context_size=self.attn_context_size,
                                  hidden_size=self.hidden_size,n_layers=self.n_dec_layers,attention=self.attention,
                                  dropout=0)




    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of 
        target sentences.

        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`

        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        return self.forward(src_sents,tgt_sents)

    def forward(self, src_sents: List[List[str]], tgt_sents: List[List[str]]) -> Tensor:
        src_encodings, decoder_init_state = self.encode(src_sents)
        scores = self.decode(src_encodings, decoder_init_state, tgt_sents)
        return scores

    def encode(self, src_sents: List[List[str]]) -> Tuple[Tensor, Any]:
        """
        Use a GRU/LSTM to encode source sentences into hidden states

        Args:
            src_sents: list of source sentence tokens

        Returns:
            src_encodings: hidden states of tokens in source sentences, this could be a variable 
                with shape (batch_size, source_sentence_length, encoding_dim), or in orther formats
            decoder_init_state: decoder GRU/LSTM's initial state, computed from source encodings
        """
        source,lens = utils.convert_to_tensor(src_sents,self.vocab.src)
#        print('input type', type(source))
        source = source.to(device)
        src_encodings,decoder_init_state = self.encoder(source,lens)
        return src_encodings, decoder_init_state

    def decode(self, src_encodings: Tensor, decoder_init_state, tgt_sents: List[List[str]]):
        """
        Given source encodings, compute the log-likelihood of predicting the gold-standard target
        sentence tokens

        Args:
            src_encodings: hidden states of tokens in source sentences
            decoder_init_state: decoder GRU/LSTM's initial state
            tgt_sents: list of gold-standard target sentences, wrapped by `<s>` and `</s>`

        Returns:
            scores: could be a variable of shape (batch_size, ) representing the 
                log-likelihood of generating the gold-standard target sentence for 
                each example in the input batch
        """
        target, lens = utils.convert_to_tensor(tgt_sents,self.vocab.tgt)
        target = target.to(device)
        batch_size, max_len = target.size()
        predictions = torch.zeros((batch_size,max_len,len(self.vocab.tgt))).to(device)
        hidden = decoder_init_state
        inputs = torch.LongTensor([self.vocab.tgt.word2id['<s>'] for _ in range(batch_size)]).to(device)
        for idx in range(0, max_len):
            outputs, hidden, attn_scores = self.decoder(inputs, hidden, src_encodings)
            inputs = target[:, idx]
            predictions[:, idx] = outputs
        return target,predictions

    def criterion(self,targets,predictions):
        batch_size, max_len, vocab_size = predictions.size()
        predictions = predictions.view(batch_size * max_len, vocab_size)
        targets = targets.view(batch_size * max_len)
        loss = self.loss(predictions, targets)
        return loss

    def beam_search_decode(self, src_encodings: Tensor, decoder_init_state, max_decoding_time_step, beam_size):
        
        def var(a):
            return Variable(a, volatile=True)

        def rvar_src(a):
            return var(a.repeat(beam_size,1, 1))

        def rvar_hidden(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(1 * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, 1, -1)




        src_encodings = rvar_src(src_encodings.data)

        # predictions = torch.zeros((beam_width, max_decoding_time_step))
        
        # hypotheses = torch.zeros((beam_width, max_decoding_time_step))
        # hidden = decoder_init_state
        # inputs = torch.LongTensor([self.vocab.tgt.word2id['<s>'] for _ in range(beam_width)])

        decState = (rvar_hidden(decoder_init_state[0].data), rvar_hidden(decoder_init_state[1].data))
        # memory = decState[0][-1]

        beam = [Beam_Class.Beam(beam_size, n_best=1)]

        # values = torch.zeros((beam_width, 1))
        
        # All the comments below inside the loop are for timesteps 1 to max_decoding_time_step. For timestep 0, the following calculations hold:
            # inputs is just ['<s>']. outputs is: (tgt_vocab_size), [no 'values' tensor gets added to outputs at idx 0 as the if condition checks that]
            # we get the values and indices as (beam_width). Now the future comments from line 204 onwards apply.

        for idx in range(0, max_decoding_time_step):

            if all((b.done() for b in beam)):
                break

            inp = var(torch.stack([b.getCurrentState() for b in beam]).t().contiguous().view(-1))


            output, decState, attn = self.decoder(inp, decState, src_encodings)

            output = unbottle(output)
            attn = unbottle(attn)

            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)
                # b.beam_update_memory(memory, j)

            # outputs, hidden, attn_scores = self.decoder(inputs, hidden, src_encodings)

            # outputs now is: (beam_width, 1, tgt_vocab). '1' in dim. 1 because we send 1 word at a time and we get the prob. dist over the vocab for the next word

            # if idx!=0:
            #     outputs = outputs.squeeze(1)
            #     values = values.unsqueeze(1)

            # outputs now is: (beam_width, tgt_vocab)
            # previous values is: (beam_width, 1)

            # Outputs + previous values broadcasted : (beam_width, tgt_vocab) [But now this also contains the log prob score of previous state it came from]
            # Now if I flattened the above summation, i will get (beam_width*tgt_vocab). Doing top-k on this will give me beam_width indices.
            
                # outputs = (outputs+values).view(-1)

            # Now, to implement length normalization, a basic approach mentioned in the reading is to divide the prob. scores by the length of the target sentence. 
            # This way, the scores we compare are the average probability per word.
            # if idx>=2:
                # multiply by (idx - 1) before dividing because in the previous iteration it would have been divided by idx-1
                # outputs = (outputs*(idx-1))/idx

            # values, indices = torch.topk(outputs, beam_width)

            # Values : Sorted Top-K probability distriubtions. Indices: their indices in tgt_vocab - this means that these indices are the word2id also.
            # Hence, the indices itself will be the input to the next timestep

            # Values and indices are both: (beam_width)

            # Now for these new indices, (indices / beam_width) will give me which previous word it was from, and (indices % beam_width) will give me which word it currently is
            # (indices % beam_width) will give me the new inputs for the next timestep. 

        
        allHyps, allScores, allAttn = [], [], []

        b = beam[0]
        n_best = 1
        scores, ks = b.sortFinished(minimum=n_best)
        hyps, attn = [], []
        for i, (times, k) in enumerate(ks[:n_best]):
            hyp, att = b.getHyp(times, k)
            hyps.append(hyp)
            attn.append(att.max(1)[1])
        allHyps.append(hyps[0])
        allScores.append(scores[0])
        allAttn.append(attn[0])

        return allHyps, allAttn 

            # indices = indices[0]
            # print (indices)
            # values = values[0]

            # if idx!=0:
            #     print(indices.shape)
            #     inputs = (indices % beam_width).unsqueeze(1)
            #     predictions[:,idx] = inputs.squeeze(1)
            #     hypotheses[:,idx] = (indices/beam_width)
            #     # hidden = hidden[hypotheses[:,idx]]
            #     print(hidden.shape)
            # else:
            #     inputs = indices
            #     predictions[:,idx] = inputs
                # hidden = (hidden[0].repeat(1,beam_width,1), hidden[1].repeat(1,beam_width,1))

            # inputs: (beam_width, 1)
            # Here, it is important that inputs is not just (beam_width), as that would mean it is a sequence with beam_width timesteps. 
            # We want the batch_size to be beam_width with each 1 input.

            # if [self.vocab.tgt.word2id['</s>']] in inputs:
            #     # hypotheses.append("<The indices recorded till now for this sentence>")
            #     beam_width-=1
            #     sentence_idx_to_stop_decoding = inputs.index([self.vocab.tgt.word2id['</s>']])
            #     inputs = inputs[:sentence_idx_to_stop_decoding] + inputs[sentence_idx_to_stop_decoding+1:]

                # Here inputs now becomes (beam_width-1, 1)

            # Now continue through the loop

            # TODO: Question...
            # (1) Now, this way, we can find the cumulative scores such that at each timestep, I have the max score possible till that timestep. I do this till min(max_decoding_time_step,</s>). 
            #       But how to keep track of the words at each timestep that are present in the beam_width?

            # (2) Implementing Length Normalization -- Implemented basic method as of now


        # The hypotheses contain the decoded indices, the values contain the log probability scores obtained for those sentences (which will be the values of the last sentence)
        # return hypotheses, values


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search

        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN

        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        
        src_encodings, decoder_init_state = self.encode([src_sent])
        scores = self.beam_search_decode(src_encodings, decoder_init_state, max_decoding_time_step, beam_size)
        return scores
        # raise NotImplementedError

    def evaluate_ppl(self, dev_data: List[Any], batch_size: int=32):
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

            target,predictions  = self.forward(src_sents, tgt_sents)
            loss = self.criterion(target,predictions)
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        vocab = pickle.load(open("data/vocab.bin", 'rb'))
        model = NMT(embed_size=int(args['--embed-size']),hidden_size=int(args['--hidden-size']),dropout_rate=float(args['--dropout']),vocab=vocab).to(device)
        # model.load_state_dict(model_path)
        model.load_state_dict(torch.load(model_path))
        return model
        # raise NotImplementedError()

    def save(self, path: str):
        """
        Save current model to file
        """

        raise NotImplementedError()


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


def train(args: Dict[str, str]):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = os.path.join(args['--save-to'], 'model_state_dict')

    vocab = pickle.load(open(args['--vocab'], 'rb'))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab).to(device)
    print('model',type(model))
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('Starting Maximum Likelihood training')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=float(args['--lr']))

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            torch.set_grad_enabled(True)
            model.train()

            train_iter += 1

            batch_size = len(src_sents)

            # (batch_size)
            targets,predictions = model(src_sents, tgt_sents)
            loss = model.criterion(targets,predictions)

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
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                torch.no_grad()
                model.eval()
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)


                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    #model.save(model_save_path)
                    torch.save(model.state_dict(),model_save_path)

                    # You may also save the optimizer's state
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model.load_state_dict(model_save_path)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

        hypotheses.append(example_hyps)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    vocab = pickle.load(open('data/vocab.bin', 'rb'))
    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            top_hyp = [vocab.tgt.id2word[int(word[0].numpy())] for word in top_hyp]
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')


def main():
    #args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
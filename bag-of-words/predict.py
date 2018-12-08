'''
 @Date  : 2017/12/28
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn
 @homepage: shumingma.com
'''

import torch
import torch.utils.data
import argparse
import time
import pickle
import os
import opts
import utils
import models
from torch.autograd import Variable
import codecs

parser = argparse.ArgumentParser(description='train.py')
#parser.add_argument('-src_file', required=True, help="input file for the data")
#parser.add_argument('-tgt_file', required=True, help="output file for the data")

opts.model_opts(parser)

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)
opts.convert_to_config(opt, config)
log_path = config.logF + opt.log + '/'

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)


def load_data():

    print('loading data...\n')
    datas = pickle.load(open('data/save_data.pkl', 'rb'))
    datas['train']['length'] = int(datas['train']['length'] * opt.scale)

    bi_testset = utils.BiDataset(datas['test'], char=config.char)

    src_vocab = datas['dict']['src']
    tgt_vocab = datas['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()

    testset = bi_testset

    test_batch_size = config.batch_size
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding)

    return {'testset': testset, 'testloader': testloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}


def eval_model(model, datas, params):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(datas['testset'])
    validloader = datas['testloader']
    tgt_vocab = datas['tgt_vocab']

    for src, tgt, src_len, tgt_len, original_src, original_tgt, data_type in validloader:

        src = Variable(src, volatile=True)
        src_len = Variable(src_len, volatile=True)
        if config.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        if config.beam_size > 1:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)
        else:
            samples, alignment = model.sample(src, src_len)

        candidate += [tgt_vocab.convertToLabels(s, utils.EOS) for s in samples]
        source += original_src
        reference += original_tgt
        if alignment is not None:
            alignments += [align for align in alignment]

        count += len(original_src)
        utils.progress_bar(count, total_count)

    if config.unk and config.attention != 'None':
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == utils.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    with codecs.open(log_path+'baseline_es_candidate.txt', 'w+', 'utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i])+'\n')

    score = {}
    for metric in config.metrics:
        score[metric] = getattr(utils, metric)(reference, candidate, params['log_path'], params['log'], config)

    return score


def build_model(checkpoints):
    # model
    print('building model...\n')
    model = getattr(models, opt.model)(config)
    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()

    return model


def build_log():
    # log
    if not os.path.exists(config.logF):
        os.mkdir(config.logF)
    if opt.log == '':
        log_path = config.logF + str(int(time.time() * 1000)) + '/'
    else:
        log_path = config.logF + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print_log = utils.print_log(log_path + 'log.txt')
    return print_log, log_path


def main():
    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n')
        checkpoints = torch.load(opt.restore)
    else:
        checkpoints = None

    datas = load_data()
    print_log, log_path = build_log()
    #model = checkpoints['model']
    model = build_model(checkpoints)

    params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']

    score = eval_model(model, datas, params)

#    for metric in config.metrics:
 #       print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))


if __name__ == '__main__':
    main()

'''
 @Date  : 2017/12/18
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''

import torch
import torch.utils.data
from torch.autograd import Variable
import lr_scheduler as L

import os
import argparse
import pickle
import time
from collections import OrderedDict

import opts
import models
import utils
import codecs

parser = argparse.ArgumentParser(description='train.py')
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
    torch.backends.cudnn.benchmark = True


def load_data():
    print('loading data...\n')
    datas = pickle.load(open('data/save_data.pkl', 'rb'))
    datas['train']['length'] = int(datas['train']['length'] * opt.scale)
    
    bi_trainset = utils.BiDataset(datas['train'], char=config.char)
    bi_validset = utils.BiDataset(datas['test'], char=config.char)

    if config.use_mono:

        mono_datas = pickle.load(open('data/save_mono_data.pkl', 'rb'))
        mono_datas['train']['length'] = int(mono_datas['train']['length'] * opt.scale)

        mono_trainset = utils.BiDataset(mono_datas['train'], data_type="mono")

    # Just ensure Src_Vocab has both mono + bi vocab together in it. For now it has only Bi vocab
    
    src_vocab = datas['dict']['src']
    tgt_vocab = datas['dict']['tgt']
    config.src_vocab_size = src_vocab.size()
    config.tgt_vocab_size = tgt_vocab.size()
    config.src_vocab = src_vocab.labelToIdx
    config.tgt_vocab = tgt_vocab.labelToIdx

    # Given that mono is the majority, most of the samples seem to be mono for a particular batch... How to circumvent this??
    if config.use_mono:
        print(" Using Monolingual and Parallel data")
        trainset = torch.utils.data.ConcatDataset([bi_trainset,mono_trainset])
    else:
        print(" Using Parallel data only")
        trainset = bi_trainset

    validset = bi_validset

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding,
                                              batch_sampler=utils.TwoDatasetBatchSampler(trainset,config.batch_size))
    '''trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding)'''
    if hasattr(config, 'valid_batch_size'):
        valid_batch_size = config.valid_batch_size
    else:
        valid_batch_size = config.batch_size
    validloader = torch.utils.data.DataLoader(dataset=validset,
                                              batch_size=valid_batch_size,
                                              shuffle=False,
                                              num_workers=0,
                                              collate_fn=utils.padding)

    return {'trainset': trainset, 'validset': validset,
            'trainloader': trainloader, 'validloader': validloader,
            'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}



def build_model(checkpoints, print_log):
#    for k, v in config.items():
#        print_log("%s:\t%s\n" % (str(k), str(v)))
    
    # model
    print('building model...\n')
    model = getattr(models, opt.model)(config)
    if checkpoints is not None:
        model.load_state_dict(checkpoints['model'])
    if opt.pretrain:
        print('loading checkpoint from %s' % opt.pretrain)
        pre_ckpt = torch.load(opt.pretrain)['model']
        pre_ckpt = OrderedDict({key[8:]: pre_ckpt[key] for key in pre_ckpt if key.startswith('encoder')})
        print(model.encoder.state_dict().keys())
        print(pre_ckpt.keys())
        model.encoder.load_state_dict(pre_ckpt)
    if use_cuda:
        model.cuda()
    
    # optimizer
    if checkpoints is not None:
        optim = checkpoints['optim']
    else:
        optim = models.Optim(config.optim, config.learning_rate, config.max_grad_norm,
                             lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optim.set_parameters(model.parameters())

    #scheduler
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

    # print log
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    for k, v in config.items():
        if type(v) == dict : continue
        print_log("%s:\t%s\n" % (str(k), str(v)))
    print_log("\n")
    print_log(repr(model) + "\n\n")
    print_log('total number of parameters: %d\n\n' % param_count)

    return model, optim, print_log


def train_model(model, datas, optim, epoch, params):

    model.train()
    trainloader = datas['trainloader']

    for src, tgt, src_len, tgt_len, original_src, original_tgt, data_type in trainloader:

        if len(data_type) != config.batch_size :
            raise Exception()

        data_type = data_type[0]

        use_xent = True if data_type == 'bi' else False
        use_label = True #if data_type == 'mono' else False

        model.zero_grad()

        src = Variable(src)
        tgt = Variable(tgt)
        src_len = Variable(src_len)
        if config.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        src = torch.index_select(src, dim=0, index=indices)
        tgt = torch.index_select(tgt, dim=0, index=indices)
        dec = tgt[:, :-1]
        targets = tgt[:, 1:]

        try:
            loss, (outputs,bow_outputs) = \
                    model(src, lengths, dec, targets, use_xent=use_xent, use_label = use_label)
            pred = torch.gather(bow_outputs.data,1,targets)
            num_correct = torch.sigmoid(pred.data).gt(0.5).masked_select(targets.ne(utils.PAD).data).sum()
            num_total = targets.ne(utils.PAD).data.sum()
            if config.max_split == 0:
                mle_loss, label_loss = loss
                params['mle_loss'] += torch.sum(mle_loss).data
                params['label_loss'] += label_loss.data
                #loss = (torch.sum(mle_loss) + min(max(0, config.alpha*(epoch-config.offset)), 1) * label_loss) / num_total.float()
                loss = (torch.sum(mle_loss) + label_loss.sum()) / num_total.float()
                loss.backward()
            optim.step()

            params['report_loss'] += loss.data
            params['report_correct'] += num_correct
            params['report_total'] += num_total

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise e

        utils.progress_bar(params['updates'], config.eval_interval)
        params['updates'] += 1

        if params['updates'] % config.eval_interval == 0:
            params['log']("epoch: %3d, loss: %6f, mle_loss: %6f, label_loss: %6f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                          % (epoch, params['report_loss'].sum(), params['mle_loss'].sum(), params['label_loss'].sum(), time.time()-params['report_time'],
                             params['updates'], params['report_correct'] * 100.0 / params['report_total']))
            print('evaluating after %d updates...\r' % params['updates'])
            score = eval_model(model, datas, params)
            for metric in config.metrics:
                params[metric].append(score[metric])
                if score[metric] >= max(params[metric]):
                    with codecs.open(log_path+'best_'+metric+'_prediction.txt','w','utf-8') as f:
                        f.write(codecs.open(log_path+'candidate.txt','r','utf-8').read())
                    save_model(params['log_path']+'best_'+metric+'_checkpoint.pt', model, optim, params['updates'])
            model.train()
            params['report_loss'], params['report_time'] = 0, time.time()
            params['report_correct'], params['report_total'] = 0, 0
            params['mle_loss'], params['label_loss'] = 0, 0

        if params['updates'] % config.save_interval == 0:
            save_model(params['log_path']+'checkpoint.pt', model, optim, params['updates'])

    #optim.updateLearningRate(score=0, epoch=epoch)


def eval_model(model, datas, params):

    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(datas['validset'])
    validloader = datas['validloader']
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

    with codecs.open(log_path+'candidate.txt','w+','utf-8') as f:
        for i in range(len(candidate)):
            f.write(" ".join(candidate[i])+'\n')

    score = {}
    for metric in config.metrics:
        score[metric] = getattr(utils, metric)(reference, candidate, params['log_path'], params['log'], config)

    return score


def save_model(path, model, optim, updates):
    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


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
    model, optim, print_log = build_model(checkpoints, print_log)
    # scheduler
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    params = {'updates': 0, 'report_loss': 0, 'report_total': 0, 'mle_loss': 0, 'label_loss': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path}
    for metric in config.metrics:
        params[metric] = []
    if opt.restore:
        params['updates'] = checkpoints['updates']

    if opt.mode == "train":
        for i in range(1, config.epoch + 1):
            if config.schedule:
                scheduler.step()
                print("Decaying learning rate to %g" % scheduler.get_lr()[0])
            train_model(model, datas, optim, i, params)
    else:
        score = eval_model(model, datas, params)

    for metric in config.metrics:
        print_log("Best %s score: %.2f\n" % (metric, max(params[metric])))


if __name__ == '__main__':
    main()

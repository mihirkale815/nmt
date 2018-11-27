'''
 @Date  : 2017/12/18
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import linecache
import torch
import torch.utils.data as torch_data
from random import Random
import utils
from torch.utils.data.sampler import Sampler
import random

num_samples = 1


class MonoDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None):

        self.srcF = infos['srcF']
        self.original_srcF = infos['original_srcF']
        self.length = infos['length']
        self.infos = infos
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()

        return src, src, original_src, original_src, "mono"
        return src, [], original_src, [], "mono"

    def __len__(self):
        return len(self.indexes)


class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False, data_type="bi"):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        self.data_type = data_type
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(self.tgtF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split() if not self.char else \
                       list(linecache.getline(self.original_tgtF, index + 1).strip())

        return src, tgt, original_src, original_tgt, self.data_type

    def __len__(self):
        return len(self.indexes)

# def get_bilingual_dict(bi_dict_path):

#     bilingual_dict = {}
#     with open(bi_dict_path,"r") as fp:
#         for line in fp:
#             line = line.strip().split()
#             bilingual_dict[line[0]] = bilingual_dict.get(line[0],[]) + [line[1]]
#     print("Bilingual Dictionary Loaded!")

#     return bilingual_dict

def splitDataset(data_set, sizes):
    length = len(data_set)
    indexes = list(range(length))
    rng = Random()
    rng.seed(1234)
    rng.shuffle(indexes)

    data_sets = []
    part_len = int(length / sizes)
    for i in range(sizes-1):
        data_sets.append(BiDataset(data_set.infos, indexes[0:part_len]))
        indexes = indexes[part_len:]
    data_sets.append(BiDataset(data_set.infos, indexes))
    return data_sets


def padding(data):
    src, tgt, original_src, original_tgt, data_type = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt, data_type


def ae_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s)[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    ae_len = [len(s)+2 for s in src]
    ae_pad = torch.zeros(len(src), max(ae_len)).long()
    for i, s in enumerate(src):
        end = ae_len[i]
        ae_pad[i, 0] = utils.BOS
        ae_pad[i, 1:end-1] = torch.LongTensor(s)[:end-2]
        ae_pad[i, end-1] = utils.EOS

    return src_pad, tgt_pad, ae_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), torch.LongTensor(ae_len), \
           original_src, original_tgt


def split_padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    split_samples = []
    num_per_sample = int(len(src) / utils.num_samples)

    for i in range(utils.num_samples):
        split_src = src[i*num_per_sample:(i+1)*num_per_sample]
        split_tgt = tgt[i*num_per_sample:(i+1)*num_per_sample]
        split_original_src = original_src[i * num_per_sample:(i + 1) * num_per_sample]
        split_original_tgt = original_tgt[i * num_per_sample:(i + 1) * num_per_sample]

        src_len = [len(s) for s in split_src]
        src_pad = torch.zeros(len(split_src), max(src_len)).long()
        for i, s in enumerate(split_src):
            end = src_len[i]
            src_pad[i, :end] = torch.LongTensor(s)[:end]

        tgt_len = [len(s) for s in split_tgt]
        tgt_pad = torch.zeros(len(split_tgt), max(tgt_len)).long()
        for i, s in enumerate(split_tgt):
            end = tgt_len[i]
            tgt_pad[i, :end] = torch.LongTensor(s)[:end]

        split_samples.append([src_pad, tgt_pad,
                              torch.LongTensor(src_len), torch.LongTensor(tgt_len),
                              split_original_src, split_original_tgt])


    return split_samples



class TwoDatasetBatchSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, batch_size,num_samples=None):
        self.data_source = data_source
        self.num_samples = num_samples
        self.batch_size = batch_size


        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        self.set_indices()
        self.drop_last = True


    def set_indices(self):
        n = len(self.data_source)
        num_batches = int(n/float(self.batch_size))
        batch_start_indices = [batch_num*self.batch_size for batch_num in range(0,num_batches)]
        random.shuffle(batch_start_indices)
        indices = [ idx for batch_start_idx in batch_start_indices for idx in range(batch_start_idx,
                                                                                    batch_start_idx+self.batch_size)]
        dataset1_size = self.data_source.cumulative_sizes[0] \
            if type(self.data_source) == torch.utils.data.ConcatDataset\
            else len(self.data_source)
        self.indices_to_ignore = set([idx for idx in range(dataset1_size - dataset1_size%self.batch_size,dataset1_size)])
        #indices = [idx for idx in indices if idx not in indices_to_ignore]
        self.indices = indices

    def __iter__(self):
        batch = []
        ctr = 0
        for idx in self.indices:
            ctr += 1
            if idx not in self.indices_to_ignore : batch.append(idx)
            if ctr == self.batch_size:
                ctr = 0
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.data_source)



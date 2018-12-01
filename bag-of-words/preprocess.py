'''
 @Date  : 2017/12/18
 @Author: Shuming Ma
 @mail  : shumingma@pku.edu.cn 
 @homepage: shumingma.com
'''
import argparse
import utils
import pickle

parser = argparse.ArgumentParser(description='preprocess.py')

parser.add_argument('-id', required=True,
                    help="id of experiment setup")
parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_filter', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_filter', type=int, default=0,
                    help="Maximum target sequence length")
parser.add_argument('-src_trun', type=int, default=0,
                    help="Truncate source sequence length")
parser.add_argument('-tgt_trun', type=int, default=0,
                    help="Truncate target sequence length")
parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")
parser.add_argument('-vocab_path', type=str, default="save_data.pkl",
                    help="path to pickle of bilingual source vocabulary")
opt = parser.parse_args()


file_config = {}

file_config['baseline'] = ['train.de-en.de', '', 'combined_train.bow.0.de-en.de', 'train.de-en.en', '',
                           'combined_train.bow.0.de-en.en','valid.de-en.de','valid.de-en.de','test.de-en.de',
                           'test.de-en.en']

file_config['bow_1'] = ['train.de-en.de', 'train.mono.1.de-en.de', 'combined_train.bow.1.de-en.de', 'train.de-en.en',
                        'train.bidict.1.de-en.en', 'combined_train.bow.1.de-en.en','valid.de-en.de','valid.de-en.en',
                        'test.de-en.de','test.de-en.en']

file_config['bow_2'] = ['train.de-en.de', 'train.mono.2.de-en.de', 'combined_train.bow.2.de-en.de', 'train.de-en.en',
                        'train.bidict.2.de-en.en', 'combined_train.bow.2.de-en.en','valid.de-en.de','valid.de-en.en',
                        'test.de-en.de','test.de-en.en']

file_config['bow_4'] = ['train.de-en.de', 'train.mono.4.de-en.de', 'combined_train.bow.4.de-en.de', 'train.de-en.en',
                        'train.bidict.4.de-en.en', 'combined_train.bow.4.de-en.en','valid.de-en.de','valid.de-en.en',
                        'test.de-en.de','test.de-en.en']

file_config['wow_1'] = ['train.de-en.de', 'train.bidict.1.de-en.de', 'combined_train.wow.1.de-en.de',
                        'train.de-en.en', 'train.mono.1.de-en.en', 'combined_train.wow.1.de-en.en','valid.de-en.de',
                        'valid.de-en.en','test.de-en.de','test.de-en.en']

file_config['wow_2'] = ['train.de-en.de', 'train.bidict.2.de-en.de', 'combined_train.wow.2.de-en.de',
                        'train.de-en.en', 'train.mono.2.de-en.en', 'combined_train.wow.2.de-en.en','valid.de-en.de',
                        'valid.de-en.en','test.de-en.de','test.de-en.en']

file_config['wow_4'] = ['train.de-en.de', 'train.bidict.4.de-en.de', 'combined_train.wow.4.de-en.de',
                        'train.de-en.en', 'train.mono.4.de-en.en', 'combined_train.wow.4.de-en.en','valid.de-en.de',
                        'valid.de-en.de','test.de-en.de','test.de-en.en']


def makeVocabulary(filename, trun_length, filter_length, char, vocab, size):

    print("%s: length limit = %d, truncate length = %d" % (filename, filter_length, trun_length))
    max_length = 0
    with open(filename, encoding='utf8') as f:
        for sent in f.readlines():
            if char:
                tokens = list(sent.strip())
            else:
                tokens = sent.strip().split()
            if 0 < filter_length < len(sent.strip().split()):
                continue
            max_length = max(max_length, len(tokens))
            if trun_length > 0:
                tokens = tokens[:trun_length]
            for word in tokens:
                vocab.add(word)

    print('Max length of %s = %d' % (filename, max_length))
    if size > 0:
        originalSize = vocab.size()
        vocab = vocab.prune(size)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)

def makeData(srcFile, tgtFile, srcDicts, tgtDicts, save_srcFile, save_tgtFile, lim=0):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')
    tgtIdF = open(save_tgtFile + '.id', 'w')
    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')
    tgtStrF = open(save_tgtFile + '.str', 'w', encoding='utf8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "" or len(sline.split())<=lim:
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()


        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter) and \
           (opt.tgt_filter == 0 or len(tline.split()) <= opt.tgt_filter):

            if opt.src_trun > 0:
                srcWords = srcWords[:opt.src_trun]
            if opt.tgt_trun > 0:
                tgtWords = tgtWords[:opt.tgt_trun]

            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)
            tgtIds = tgtDicts.convertToIdx(tgtWords, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD)

            srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')
            tgtIdF.write(" ".join(list(map(str, tgtIds)))+'\n')

            srcStrF.write(" ".join(srcWords)+'\n')
            tgtStrF.write(" ".join(tgtWords)+'\n')



            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    srcStrF.close()
    tgtStrF.close()
    srcIdF.close()
    tgtIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': save_tgtFile + '.id',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': save_tgtFile + '.str',
            'length': sizes}

def mono_makeData(srcFile, srcDicts, save_srcFile, lim=0):
    sizes = 0
    count, empty_ignored, limit_ignored = 0, 0, 0

    print('Processing %s ...' % (srcFile))
    srcF = open(srcFile, encoding='utf8')

    srcIdF = open(save_srcFile + '.id', 'w')

    srcStrF = open(save_srcFile + '.str', 'w', encoding='utf8')


    while True:
        sline = srcF.readline()


        # normal end of file
        if sline == "":
            break

        sline = sline.strip()

        # source and/or target are empty
        if sline == "" or len(sline.split())<=lim:
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            empty_ignored += 1
            continue

        sline = sline.lower()

        srcWords = sline.split()


        if (opt.src_filter == 0 or len(sline.split()) <= opt.src_filter):

            if opt.src_trun > 0:
                srcWords = srcWords[:opt.src_trun]

            srcIds = srcDicts.convertToIdx(srcWords, utils.UNK_WORD)

            srcIdF.write(" ".join(list(map(str, srcIds)))+'\n')

            srcStrF.write(" ".join(srcWords)+'\n')


            sizes += 1
        else:
            limit_ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    srcStrF.close()
    srcIdF.close()

    print('Prepared %d sentences (%d and %d ignored due to length == 0 or > )' %
          (sizes, empty_ignored, limit_ignored))

    return {'srcF': save_srcFile + '.id', 'tgtF': '',
            'original_srcF': save_srcFile + '.str', 'original_tgtF': '',
            'length': sizes}


def main():

    dicts = {}

    train_src, train_mono_src, combined_train_src, \
    train_tgt, train_mono_tgt, combined_train_tgt,  \
    valid_src, valid_tgt, \
    test_src, test_tgt  = file_config[opt.id]

    train_src = "data/" + train_src
    train_mono_src = "data/" + train_mono_src

    train_tgt = "data/" + train_tgt
    train_mono_tgt = "data/" + train_mono_tgt

    combined_train_src = "data/" + combined_train_src
    combined_train_tgt = "data/" + combined_train_tgt

    valid_src = "data/" + valid_src
    valid_tgt = "data/" + valid_tgt

    test_src = "data/" + test_src
    test_tgt = "data/" + test_tgt

    save_train_src, save_train_tgt = train_src + ".save", train_tgt + ".save"
    save_valid_src, save_valid_tgt = valid_src + ".save", valid_tgt + ".save"
    save_test_src, save_test_tgt = test_src + ".save", test_tgt + ".save"
    src_dict, tgt_dict = "save" + '.src.dict', "save" + '.tgt.dict'




    print('Building source vocabulary...')
    dicts['src'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
    dicts['src'] = makeVocabulary(combined_train_src, opt.src_trun, opt.src_filter, False, dicts['src'], opt.src_vocab_size)
    print('Building target vocabulary...')
    dicts['tgt'] = utils.Dict([utils.PAD_WORD, utils.UNK_WORD, utils.BOS_WORD, utils.EOS_WORD])
    dicts['tgt'] = makeVocabulary(combined_train_tgt, opt.tgt_trun, opt.tgt_filter, False, dicts['tgt'], opt.tgt_vocab_size)


    print('Preparing training ...')
    train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt)

    print('Preparing validation ...')
    valid = makeData(valid_src, valid_tgt, dicts['src'], dicts['tgt'], save_valid_src, save_valid_tgt)

    print('Preparing test ...')
    test = makeData(test_src, test_tgt, dicts['src'], dicts['tgt'], save_test_src, save_test_tgt)

    print('Saving source vocabulary to \'' + src_dict + '\'...')
    dicts['src'].writeFile(src_dict)

    print('Saving target vocabulary to \'' + tgt_dict + '\'...')
    dicts['tgt'].writeFile(tgt_dict)

    datas = {'train': train, 'valid': valid,
             'test': test, 'dict': dicts}
    pickle.dump(datas, open('data/save_data.pkl', 'wb'))

    if train_mono_src!= ''  and train_mono_tgt!= '' :
        print('Preparing mono training ...')

        # Getting the vocabulary from Bi-dataset
        datas = pickle.load(open("data/"+opt.vocab_path, 'rb'))
        dicts['src'] = datas['dict']['src']
        dicts['tgt'] = datas['dict']['tgt']

        train = makeData(train_src, train_tgt, dicts['src'], dicts['tgt'], save_train_src, save_train_tgt)
    
        datas = {'train': train, 'valid': {},
                 'test': {}, 'dict': dicts}
        pickle.dump(datas, open('data/save_mono_data.pkl', 'wb'))


if __name__ == "__main__":
    main()

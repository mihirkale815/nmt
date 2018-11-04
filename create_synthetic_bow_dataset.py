from collections import defaultdict



def get_word_mappings(bilingual_dict_path):
    src2tgt = defaultdict(list)
    f = open(bilingual_dict_path)
    for line in f:
        src_word,tgt_word,prob = line.strip("\n").split()
        prob = float(prob)
        src2tgt[src_word].append((tgt_word,prob))
    return src2tgt

def get_best_mapping(src2tgt):
    for src_word,items in src2tgt.items():
        tgt_word = sorted(items,key=lambda x : x[1], reverse=True)
        src2tgt[src_word] = tgt_word
    return src2tgt


if __name__=='__main__':

    bilingual_dict_path = ""
    monolingual_data_path = ""
    src_path = ""
    tgt_path = ""

    src2tgts = get_word_mappings(bilingual_dict_path)
    src2tgt = get_best_mapping(src2tgts)


    f_mono = open(monolingual_data_path)
    f_src = open(src_path,"w")
    f_tgt = open(tgt_path,"w")

    for line in f_mono:
        src_words = line.strip("\n").split()
        tgt_words = [src2tgt[word] for word in src_words]
        f_src.write(line)
        f_tgt.write(" ".join(tgt_words) + "\n")
    f_mono.close()
    f_tgt.close()
    f_src.close()


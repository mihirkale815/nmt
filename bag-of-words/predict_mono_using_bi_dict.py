import sys, numpy as np

def get_bilingual_dict(bi_dict_path):

    bilingual_dict = {}
    with open(bi_dict_path,"r") as fp:
        for line in fp:
            line = line.strip().split()
            bilingual_dict[line[0]] = bilingual_dict.get(line[0],[]) + [line[1]]
    print("Bilingual Dictionary Loaded!")

    return bilingual_dict

def predict_mono(mono_file_path, bi_dict, out_tgt_file):

	with open(mono_file_path,"r") as fr, open(out_tgt_file, "w") as fs:
		for line in fr:
			line = line.split()
			translated_line = [np.random.choice(bi_dict.get(word)) for word in line if word in bi_dict]
			fs.write(' '.join(translated_line)+'\n')

if __name__ == "__main__":

	mono_file_path = sys.argv[1]
	bi_dict_path = sys.argv[2]
	out_tgt_file = sys.argv[3]

	bi_dict = get_bilingual_dict(bi_dict_path)
	predict_mono(mono_file_path, bi_dict, out_tgt_file)




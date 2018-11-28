import sys, numpy as np

def get_bilingual_dict(bi_dict_path):
	bilingual_dict,rev_bilingual_dict = {},{}
	with open(bi_dict_path,"r") as fp:
		for line in fp:
			line = line.strip().split()
			bilingual_dict[line[0]] = bilingual_dict.get(line[0],[]) + [line[1]]
			rev_bilingual_dict[line[1]] = rev_bilingual_dict.get(line[1], []) + [line[0]]
	print("Bilingual Dictionary Loaded!")
	return bilingual_dict,rev_bilingual_dict

def predict_mono(mono_file_path, bi_dict, out_tgt_file):

	with open(mono_file_path,"r") as fr, open(out_tgt_file, "w") as fs:
		for line in fr:
			line = line.split()
			translated_line = [np.random.choice(bi_dict.get(word)) for word in line if word in bi_dict]
			fs.write(' '.join(translated_line)+'\n')

if __name__ == "__main__":

	lang_pair = sys.argv[1]
	src_suf,tgt_suf = lang_pair.split("-")
	bi_dict_path = lang_pair + ".txt"

	bi_dict,rev_bi_dict = get_bilingual_dict(bi_dict_path)

	for mult in (1,2,4):
		mono_file_path = ".".join(["train","mono",str(mult),lang_pair,src_suf])
		out_tgt_file = ".".join(["train", "bidict",str(mult), lang_pair, tgt_suf])
		predict_mono(mono_file_path, bi_dict, out_tgt_file)

		mono_file_path = ".".join(["train", "mono", str(mult), lang_pair, tgt_suf])
		out_tgt_file = ".".join(["train", "bidict", str(mult), lang_pair, src_suf])
		predict_mono(mono_file_path, rev_bi_dict, out_tgt_file)




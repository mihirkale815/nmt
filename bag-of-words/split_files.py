import sys
import random


def write_lines_tp_file(lines,path):
	fp = open(path, "w")
	for line in lines:
		fp.write(line)
	fp.close()



src_filename = sys.argv[1]
tgt_filename = sys.argv[2]
lang_pair = sys.argv[3]
max_lines = int(sys.argv[4])

data = []
src_fp = open(src_filename, "r")
tgt_fp = open(tgt_filename, "r")

for line in src_fp:
	data.append([line,next(tgt_fp)])
src_fp.close()
tgt_fp.close()

random.shuffle(data)

parallel_data = data[0:max_lines]
monolingual_data = data[max_lines:]


src_lang,tgt_lang = lang_pair.split("-")
src_save_file = ".".join(["train",lang_pair,src_lang])
tgt_save_file = ".".join(["train",lang_pair,tgt_lang])

write_lines_tp_file([d[0] for d in parallel_data],src_save_file)
write_lines_tp_file([d[1] for d in parallel_data],tgt_save_file)


for mult in (1,2,4):
	data = monolingual_data[0:mult*max_lines]
	src_mono_save_file = ".".join(["train",'mono',str(mult),lang_pair,src_lang])
	tgt_mono_save_file = ".".join(["train", 'mono', str(mult), lang_pair, tgt_lang])
	write_lines_tp_file([d[0] for d in data], src_mono_save_file)
	write_lines_tp_file([d[1] for d in data], tgt_mono_save_file)



				
		
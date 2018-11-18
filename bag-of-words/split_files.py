import sys

filename = sys.argv[1]
save_file = sys.argv[2]
mono_save_file = sys.argv[3]
max_lines = int(sys.argv[4])

count = 0

with open(filename, "r") as fp, open(save_file, "w") as fs, open(mono_save_file, "w") as fm:
		for line in fp:
			count+=1
			if count > max_lines:
				fm.write(line)
			else:
				fs.write(line)
				
		
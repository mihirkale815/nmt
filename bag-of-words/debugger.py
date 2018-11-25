count=0
with open("data/train.mono_de-en.en","r") as fp:
	for line in fp:
		count+=1
		if count>=7370:
			print("LINE:")
			print(line)
			s = input()
		print(count)
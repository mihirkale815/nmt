echo "Splitting source data into monolingual and subset for parallel corpora"
cd data
python ../split_files.py original_train.de-en.de  original_train.de-en.en   de-en 10000

python ../predict_mono_using_bi_dict.py de-en

cat train.de-en.de > combined_train.bow.0.de-en.de
cat train.de-en.en  > combined_train.bow.0.de-en.en

cat train.de-en.de train.mono.1.de-en.de > combined_train.bow.1.de-en.de
cat train.de-en.en train.bidict.1.de-en.en > combined_train.bow.1.de-en.en

cat train.de-en.de train.mono.2.de-en.de > combined_train.bow.2.de-en.de
cat train.de-en.en train.bidict.2.de-en.en > combined_train.bow.2.de-en.en

cat train.de-en.de train.mono.4.de-en.de > combined_train.bow.4.de-en.de
cat train.de-en.en train.bidict.4.de-en.en > combined_train.bow.4.de-en.en

cat train.de-en.de train.bidict.1.de-en.de > combined_train.wow.1.de-en.de
cat train.de-en.en train.mono.1.de-en.en > combined_train.wow.1.de-en.en

cat train.de-en.de train.bidict.2.de-en.de > combined_train.wow.2.de-en.de
cat train.de-en.en train.mono.2.de-en.en > combined_train.wow.2.de-en.en

cat train.de-en.de train.bidict.4.de-en.de > combined_train.wow.4.de-en.de
cat train.de-en.en train.mono.4.de-en.en > combined_train.wow.4.de-en.en



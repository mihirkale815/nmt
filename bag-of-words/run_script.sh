echo "Splitting source data into monolingual and subset for parallel corpora"

language_folder="de-en"

cd data/$language_folder

echo "$language_folder"

python ../../split_files.py original_train.src-tgt.src  original_train.src-tgt.tgt src-tgt 10000

python ../../predict_mono_using_bi_dict.py src-tgt

cat train.src-tgt.src > combined_train.bow.0.src-tgt.src
cat train.src-tgt.tgt  > combined_train.bow.0.src-tgt.tgt

cat train.src-tgt.src train.mono.1.src-tgt.src > combined_train.bow.1.src-tgt.src
cat train.src-tgt.tgt train.bidict.1.src-tgt.tgt > combined_train.bow.1.src-tgt.tgt

cat train.src-tgt.src train.mono.2.src-tgt.src > combined_train.bow.2.src-tgt.src
cat train.src-tgt.tgt train.bidict.2.src-tgt.tgt > combined_train.bow.2.src-tgt.tgt

cat train.src-tgt.src train.mono.4.src-tgt.src > combined_train.bow.4.src-tgt.src
cat train.src-tgt.tgt train.bidict.4.src-tgt.tgt > combined_train.bow.4.src-tgt.tgt

cat train.src-tgt.src train.bidict.1.src-tgt.src > combined_train.wow.1.src-tgt.src
cat train.src-tgt.tgt train.mono.1.src-tgt.tgt > combined_train.wow.1.src-tgt.tgt

cat train.src-tgt.src train.bidict.2.src-tgt.src > combined_train.wow.2.src-tgt.src
cat train.src-tgt.tgt train.mono.2.src-tgt.tgt > combined_train.wow.2.src-tgt.tgt

cat train.src-tgt.src train.bidict.4.src-tgt.src > combined_train.wow.4.src-tgt.src
cat train.src-tgt.tgt train.mono.4.src-tgt.tgt > combined_train.wow.4.src-tgt.tgt



echo "Splitting source data into monolingual and subset for parallel corpora"
python ../split_files.py original_train.de-en.de  original_train.de-en.en   de-en 10000

python predict_mono_using_bi_dict.py de-en

cat train.de-en.de train.mono.1.de-en.de > combined_train.1.de-en.de
cat train.de-en.en train.bidict.1.de-en.en > combined_train.1.de-en.en

echo "Create Vocabulary, Train, Test, Valid sets and dump as pickle"
python preprocess.py \
	-mono=0 \
	-data_folder="data/" \
	-save_data="de-en_save" \
	-load_data="de-en" \
	-src_suf="de" \
	-tgt_suf="en" \
	-report_every=1000 \
	-mult 1


echo "Create Train set and dump as pickle for mono"
python preprocess.py \
	-mono=1 \
	-data_folder="data/" \
	-save_data="de-en_save_mono" \
	-vocab_path="data/de-en_savedata.pkl" \
	-load_data="mono.1.de-en" \
	-src_suf="de" \
	-tgt_suf="en" \
	-report_every=1000 \
	-mult 1

echo "Starting Main Training Loop"

python train.py \
	-config="fr_en.yaml" \
	-mode="train" \
    #    -gpus=0 \


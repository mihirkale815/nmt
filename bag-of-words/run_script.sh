#echo "Splitting source data into monolingual and subset for parallel corpora"
#python split_files.py data/train.en-gl.gl data/train.de-en.de data/train.mono_de-en.de 10000

#echo "Splitting target data only into corresponding parallel corpora component as source data"
#python split_files.py data/train.en-gl.en data/train.de-en.en data/train.mono_de-en.en 10000

echo "Create Vocabulary, Train, Test, Valid sets and dump as pickle"
python preprocess.py \
	-mono=1 \
	-data_folder="data/" \
	-save_data="en-gl_save" \
	-load_data="en-gl" \
	-src_suf="gl" \
	-tgt_suf="en" \
	-report_every=1000

#echo "Create Train set and dump as pickle for mono"
#python preprocess.py \
#	-mono=1 \
#	-data_folder="data/" \
#	-save_data="de-en_save_mono" \
#	-vocab_path="data/de-en_savedata.pkl" \
#	-load_data="mono_de-en" \
#	-src_suf="de" \
#	-tgt_suf="en" \
#	-report_every=1000

echo "Starting Main Training Loop"

python train.py \
	-config="fr_en.yaml" \
	-mode="train" \
    #    -gpus=0 \


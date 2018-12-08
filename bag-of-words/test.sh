python preprocess.py -id="baseline_20_es"  -report_every=1000

echo "Starting Testing Loop"

python predict.py -config="fr_en_test.yaml" -gpus=0 -restore='/home/ubuntu/srey-nmt/bag-of-words/saved_models/es_en_baseline_1/best_bleu_checkpoint.pt'

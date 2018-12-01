python preprocess.py -id="test_mono"  -report_every=1000

echo "Starting Testing Loop"

python predict.py -config="fr_en.yaml" -gpus=0 -restore='/home/sreyashn/nmt/bag-of-words/experiments/1543559316339/best_bleu_checkpoint.pt'
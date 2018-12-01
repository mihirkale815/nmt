python preprocess.py -id="baseline"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="fr_en.yaml" -mode="train"  -gpus=0


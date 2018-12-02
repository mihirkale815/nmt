python preprocess.py -id="bow_2" -data_folder="de-en"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="de-en.yaml" -mode="train"  -gpus=0


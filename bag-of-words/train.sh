python preprocess.py -id="baseline" -data_folder="es-en"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="es_en_baseline.yaml" -mode="train"  -gpus=0


python preprocess.py -id="wow_1" -data_folder="es-en"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="es_en_wow1.yaml" -mode="train"  -gpus=0


python preprocess.py -id="wow_2" -data_folder="es-en"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="es_en_wow2.yaml" -mode="train"  -gpus=0


python preprocess.py -id="wow_4" -data_folder="es-en"  -report_every=1000

echo "Starting Main Training Loop"

python train.py -config="es_en_wow4.yaml" -mode="train"  -gpus=0
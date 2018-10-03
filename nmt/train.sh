#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

# python nmt.py \
#     train \
#     --cuda \
#     --vocab ${vocab} \
#     --train-src ${train_src} \
#     --train-tgt ${train_tgt} \
#     --dev-src ${dev_src} \
#     --dev-tgt ${dev_tgt} \
#     --save-to ${work_dir}/model.bin \
#     --valid-niter 2400 \
#     --batch-size 64 \
#     --hidden-size 256 \
#     --embed-size 256 \
#     --uniform-init 0.1 \
#     --dropout 0.2 \
#     --clip-grad 5.0 \
#     --lr-decay 0.5 2>${work_dir}/err.log

python nmt.py train \
            --train-src='data/train.de-en.en.wmixerprep' \
            --train-tgt='data/train.de-en.de.wmixerprep' \
            --dev-src='data/valid.de-en.en.wmixerprep' \
            --dev-tgt='data/valid.de-en.de.wmixerprep' \
            --vocab='data/vocab.bin' \
            --seed=0 \
            --batch-size=16 \
            --embed-size=128 \
            --hidden-size=128 \
            --valid-niter=50 \
            --clip-grad=5 \
            --log-every=10 \
            --max-epoch=30 \
            --patience=5 \
            --max-num-trial=5 \
            --lr-decay=0.5 \
            --beam-size=5 \
            --lr=0.001 \
            --uniform-init=0.1 \
            --save-to=experiments \
            --dropout=0.2 \
            --max-decoding-time-step=70

python nmt.py \
    decode \
    --max-decoding-time-step=100 \
    --embed-size=128 \
    --hidden-size=128 \
    --beam-size=5 \
    --dropout=0.2 \
    experiments/model_state_dict \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
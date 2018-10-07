#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"

echo save results to ${work_dir}



python nmt.py \
    decode \
    --max-decoding-time-step=100 \
    --embed-size=256 \
    --hidden-size=256 \
    --beam-size=5 \
    --dropout=0.2 \
    experiments/model_state_dict \
    ${test_src} \
    ${test_tgt} \
    ${work_dir}/decode_bleh.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode_bleh.txt
#!/bin/sh

vocab="data/vocab.bin"
# train_src="data/train.de-en.de.wmixerprep"
# train_tgt="data/train.de-en.en.wmixerprep"
# dev_src="data/valid.de-en.de"
# dev_tgt="data/valid.de-en.en"
test_src="data/test.en-gl.gl.txt"
test_tgt="data/test.en-gl.en.txt"

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
            --lr_train-src='data/train.en-gl.gl.txt' \
            --hr_train-src='data/train.en-pt.pt.txt' \
            --lr_train-tgt='data/train.en-gl.en.txt' \
            --hr_train-tgt='data/train.en-pt.en.txt' \
            --lr_dev-src='data/dev.en-gl.gl.txt' \
            --hr_dev-src='data/dev.en-pt.pt.txt' \
            --lr_dev-tgt='data/dev.en-gl.en.txt' \
            --hr_dev-tgt='data/dev.en-pt.en.txt' \
            --vocab='data/vocab.bin' \
            --seed=0 \
            --batch-size=64 \
            --embed-size=256 \
            --hidden-size=256 \
            --valid-niter=2400 \
            --clip-grad=5 \
            --log-every=50 \
            --max-epoch=30 \
            --patience=5 \
            --max-num-trial=5 \
            --lr-decay=0.5 \
            --beam-size=5 \
            --lr=0.001 \
            --uniform-init=0.1 \
            --save-to=experiments \
            --dropout=0.2 \
            --max-decoding-time-step=70 \
            --low_resource_penalty=0.7 \
            --valid_on_low_resource=True


# python nmt.py \
#     decode \
#     --max-decoding-time-step=100 \
#     --embed-size=256 \
#     --hidden-size=256 \
#     --beam-size=5 \
#     --dropout=0.2 \
#     --cuda
#     experiments/model_state_dict \
#     ${test_src} \
#     ${work_dir}/decode_test.txt

# perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode_test.txt

#!/bin/sh

vocab="data/vocab.bin"
train_src='data/train.en-gl.gl.txt'
train_tgt='data/train.en-gl.en.txt' 
dev_src='data/dev.en-gl.gl.txt'
dev_tgt='data/dev.en-gl.en.txt'
test_src='data/test.en-gl.gl.txt'
test_tgt='data/test.en-gl.en.txt'

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
            --train-src='data/train.en-gl.gl.txt' \
            --train-tgt='data/train.en-gl.en.txt' \
            --dev-src='data/dev.en-gl.gl.txt' \
            --dev-tgt='data/dev.en-gl.en.txt' \
            --vocab='data/vocab.bin' \
            --test-src='data/test.en-gl.gl.txt' \
            --test-tgt='data/test.en-gl.en.txt' \
            --beam_size=5 \
            --model_path=experiments/model_state_dict \
            --out-file=${work_dir}\
            --seed=0 \
            --batch-size=8 \
            --embed-size=128 \
            --hidden-size=128 \
            --valid-niter=640 \
            --clip-grad=5 \
            --log-every=10 \
            --max-epoch=5 \
            --patience=5 \
            --max-num-trial=5 \
            --lr-decay=0.5 \
            --lr=0.001 \
            --uniform-init=0.1 \
            --save-to=experiments \
            --dropout=0.2 \
            --max-decoding-time-step=70 \
            --cuda 

           

#python nmt.py \
 #   decode \
  #  --max-decoding-time-step=100 \
   # --embed-size=256 \
 #   --hidden-size=256 \
  #  --beam-size=5 \
#    --dropout=0.2 \
 #   --cuda
 #   experiments/model_state_dict \
  #  ${test_src} \
  #  ${work_dir}/decode_2layer.txt

#perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode_2layer.txt

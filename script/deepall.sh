#!/bin/bash

domain=("photo" "art" "cartoon" "sketch")

times=5
for i in `seq 1 $times`
do
    max=$((${#domain[@]}-1))
    for j in `seq 0 $max`
    do
        dir_name="PACS/default/${domain[j]}${i}"
        echo $dir_name
        python ../main/main.py \
        --data-root='/data/unagi0/matsuura/PACS/raw_images/kfold/' \
        --save-root='/data/unagi0/matsuura/result/dg_mmld/' \
        --result-dir=$dir_name \
        --train='deepall' \
        --data='PACS' \
        --model='caffenet' \
        --exp-num=$j \
        --gpu=0 \
        --num-epoch=30 \
        --scheduler='step' \
        --lr=1e-3 \
        --lr-step=24 \
        --lr-decay-gamma=0.1 \
        --nesterov \
        --fc-weight=10.0 \
        --color-jitter \
        --min-scale=0.8
    done
done
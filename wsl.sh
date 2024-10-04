#! /bin/bash

. ./path.sh

config=./conf.ini
iter=3
model_dir=results/model_$iter

python -B train_wsl.py --epoch 35 --lr 0.0001 --batch_size 2 --optim_step_size 20 30 35\
                         --img $iter_dir/rgb_3 --csv $csv_dir --output $model_dir --pretrain $prt_conv --conf $config || exit 1;


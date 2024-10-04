#! /bin/bash

. ./path.sh
config=./conf.ini

model_dir=results/model_3
python -B train_supcon.py --epoch 20 --batch_size 256 --optim_step_size 20 --lr 0.0001 --img $iter_dir/rgb_3 --csv $csv_dir --output $model_dir --conf $config || exit 1;
#! /bin/bash

. ./path.sh
parts=(train dev test)
config=./conf.ini
iter=3
model_dir=results/model_3
partition=test

python -B evaluate.py --model_pth $model_dir/best.pth --img $iter_dir/rgb_3 --csv $csv_dir --partition $partition --lm_pth $lm_dir/best.pth --conf $config || exit 1;


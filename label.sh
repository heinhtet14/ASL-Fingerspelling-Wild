#! /bin/bash

. ./path.sh
parts=(train dev test)
config=./conf.ini

model_dir=results/model_3
python -B make_label.py --output_path results/model_3 --model_pth $model_dir/best.pth --img $iter_dir/rgb_3 --csv $csv_dir --conf $config || exit 1;
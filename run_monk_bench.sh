#!/bin/bash

pip install -r requirements.txt

python3 monk-bench.py \
    --bench_name monks-3 \
    --task classification \
    --lr 0.1 \
    --momentum 0.0 \
    --reg 0.0 \
    --max_epochs 1000 \
    --sigma 0.01 \
    --mu 0.0 \
    --layers 17 128 1 \
    --activation Linear \
    --activation_last_layer Sigmoid \
    --verbose yes \
    --debug_interval 100

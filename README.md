# ml-lib
Yet another light Python library to build, train and validate Multi-Layer Perceptrons (MLP) with a few lines of code.
ml-lib implements from scratch the backpropagation algorithm, and a parallel model selection through k​-​cross validation.
As a benchmark, trained MLPs match state-of-the-art accuracy on “The Monk’s Problems” benchmark dataset.

## Overview

* [nn/](nn) contains all the underlying code needed to train MLPs and for doing model selection through grid-search of hyper-parameters space, and using k-cross validation.
* [monks-data/](monks-data) contains benchmark data from The Monk's Problem dataset.
* [utils/](utils) contains [utils/plot](utils/plot) with code used to plot data as a result of training phase.

To start out with the benchmark, [run_monk_bench.sh](run_monk_bench.sh) contains the bash script to run the benchmark on The Monk's Problem dataset.

## Usage

We assume that you're using [Python 3.7+](https://www.python.org/downloads/) with [pip](https://pip.pypa.io/en/stable/installing/) installed.

To run the benchmark, do

```bash
./run_monk_bench.sh
```

it will also install all required packages from ```requirements.txt```.

Take a look at the commands defined in [run_monk_bench.sh](run_monk_bench.sh) to change the monks' task, e.g. monks-1, monks-2 or monks-3.

```bash
python monk-bench.py \
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
```

If you also intend to run the model selection enable it through command ```--do_grid_search```. Also, if you prefer to add heuristics to regularize training with early-stopping, enable it with command ```--do_early_stopping```, its implementation takes inspiration from the paper [Early Stopping - but when?](https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf).
#!/bin/bash

experiments="
2014,2015,2016
2014,2015,2017
2014,2015,2018
2014,2016,2017
2014,2016,2018
2014,2017,2018
2015,2016,2017
2015,2016,2018
2015,2017,2018
2016,2017,2018
"

mkdir -p results

for e in $experiments
do
    echo running $e
    python main.py --hidden_dim=128 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=1000 --irm_penalty_weight=0 --rex_penalty_weight=1e4 --steps=9001 --n_restarts=3 --save=rex --train_envs=$e

    python main.py --hidden_dim=128 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=1000 --irm_penalty_weight=1e4 --rex_penalty_weight=0 --steps=9001 --n_restarts=3 --save=irm --train_envs=$e

    python main.py --hidden_dim=128 --l2_regularizer_weight=0.001 --lr=0.001 --penalty_anneal_iters=1000 --irm_penalty_weight=0 --rex_penalty_weight=1e4 --steps=9001 --n_restarts=3 --save=erm --train_envs=$e
done


python plot_grid.py irm
python plot_grid.py rex
python plot_grid.py erm

python plot_grid.py all

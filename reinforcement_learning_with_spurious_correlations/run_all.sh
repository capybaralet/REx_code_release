#!/bin/bash

NOW=$(date +"%m%d%H%M")

# ./run_cluster.sh cartpole_swingup
./run_cluster.sh reacher_easy
# ./run_cluster.sh cheetah_run
./run_cluster.sh finger_spin
# ./run_cluster.sh ball_in_cup_catch 
./run_cluster.sh walker_walk
./run_cluster.sh walker_stand
# ./run_cluster.sh walker_run


# cartpole_balance
# cartpole_balance_sparse
# cup_catch
# reacher_hard
# finger_turn_easy
# finger_turn_hard
# cheetah_run
# reacher_easy
# finger_spin
# cartpole_swingup_sparse
# hopper_stand
# hopper_hop
# quadruped_walk
# quadruped_run
# pendulum_swingup
# acrobot_swingup
# walker_stand
# walker_walk
# walker_run

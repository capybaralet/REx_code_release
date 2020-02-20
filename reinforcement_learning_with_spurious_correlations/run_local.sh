#!/bin/bash

DOMAIN=cartpole
TASK=swingup

SAVEDIR=./save

CUDA_VISIBLE_DEVICES=1 python train.py \
    env=${DOMAIN}_${TASK} \
    agent=rrex \
    experiment=${DOMAIN}_${TASK}_rex \
    agent.params.penalty_type=rex \
    seed=1

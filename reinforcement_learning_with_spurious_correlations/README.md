# Soft Actor-Critic (SAC) implementation in PyTorch with REx

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment and activate it:
```
conda env create -f conda_env.yml
source activate pytorch_sac
```

## Instructions
To train an SAC-REx agent on the `cartpole swingup` task run:
```
./run_local.sh
```
This will produce `exp` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir exp
```

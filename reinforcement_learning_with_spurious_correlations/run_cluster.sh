#!/bin/bash

CURDIR=`pwd`
CODEDIR=`mktemp -d -p ${CURDIR}/tmp`

cp ${CURDIR}/*.py ${CODEDIR}
cp -r ${CURDIR}/dmc2gym ${CODEDIR}/
cp -r ${CURDIR}/config ${CODEDIR}/
cp -r ${CURDIR}/agent ${CODEDIR}/

ENV=${1:-cartpole_swingup}
NOW=${2:-$(date +"%m%d%H%M")}
NOISE=0.01
NOISE_DIM=1

CDIR=/checkpoint/${USER}/rrex/${ENV}
mkdir -p ${CDIR}

for AGENT in 'rrex'; do
for PENALTY_TYPE in  'irm'; do
for ANNEAL_ITERS in 20000; do
for PENALTY in 1; do
for SEED in 1 2 3 4 5 6 7 8 9 10; do
  EXPERIMENT=${PENALTY_TYPE}_penalty${PENALTY}_evalfactor3_2envs_criticbackprop_noise${NOISE}dim${NOISE_DIM}_rewardloss/seed_${SEED}
  SAVEDIR=${CDIR}/${AGENT}_${EXPERIMENT}
  mkdir -p ${SAVEDIR}
  JOBNAME=rrex_${NOW}_${ENV}
  SCRIPT=${SAVEDIR}/run.sh
  SLURM=${SAVEDIR}/run.slrm
  CODEREF=${SAVEDIR}/code
  extra=""
  echo "#!/bin/sh" > ${SCRIPT}
  echo "#!/bin/sh" > ${SLURM}
  echo ${CODEDIR} > ${CODEREF}
  echo "#SBATCH --job-name=${JOBNAME}" >> ${SLURM}
  echo "#SBATCH --output=${SAVEDIR}/stdout" >> ${SLURM}
  echo "#SBATCH --error=${SAVEDIR}/stderr" >> ${SLURM}
  echo "#SBATCH --partition=learnfair" >> ${SLURM}
  echo "#SBATCH --comment='ICML deadline'" >> ${SLURM}
  echo "#SBATCH --nodes=1" >> ${SLURM}
  echo "#SBATCH --time=4000" >> ${SLURM}
  echo "#SBATCH --ntasks-per-node=1" >> ${SLURM}
  echo "#SBATCH --signal=USR1" >> ${SLURM}
  echo "#SBATCH --gres=gpu:volta:1" >> ${SLURM}
  echo "#SBATCH --mem=150000" >> ${SLURM}
  echo "#SBATCH -c 1" >> ${SLURM}
  echo "srun sh ${SCRIPT}" >> ${SLURM}
  echo "echo \$SLURM_JOB_ID >> ${SAVEDIR}/id" >> ${SCRIPT}
  echo "nvidia-smi" >> ${SCRIPT}
  echo "cd ${CODEDIR}" >> ${SCRIPT}
  echo MUJOCO_GL="osmesa" python train.py \
    env=${ENV} \
    agent=${AGENT} \
    experiment=${EXPERIMENT} \
    seed=${SEED} \
    noise=${NOISE} \
    noise_dims=${NOISE_DIM} \
    agent.params.penalty_type=${PENALTY_TYPE} \
    agent.params.penalty_weight=${PENALTY} \
    num_train_encoder_steps=5000 \
    agent.params.penalty_anneal_iters=${ANNEAL_ITERS} >> ${SCRIPT}
  sbatch ${SLURM}
done
done
done
done
done
#!/bin/bash
#SBATCH --job-name=original_hrviton
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -w mscluster107
#SBATCH -t 72:00:00
#SBATCH --output="/home-mscluster/mmolefe/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/original_hrviton/original.hrviton.%N.%j.out"
#SBATCH --ntasks=1
#SBATCH --partition=biggpu
#SBATCH -e "/home-mscluster/mmolefe/Playground/Synthesising Virtual Fashion Try-On with Neural Radiance Fields/Fashion-SuperNeRF/original_hrviton/original.hrviton.%N.%j.err"


echo ------------------------------------------------------
echo -n 'Job is running on node ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of cores is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------

/bin/hostname
nvidia-smi
source ~/.bashrc
conda activate NeRF
python3 train_condition.py --cuda True --gpu_ids 0 --Ddownx2 --Ddropout --lasttvonly --interflowloss

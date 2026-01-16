#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate diff && \
cd /mnt/petrelfs/zhangsiyu/Diff/DDPM && \
python train.py
"

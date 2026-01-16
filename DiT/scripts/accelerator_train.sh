#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:4

srun -p $vp --gres=gpu:4 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate diff && \
cd /mnt/petrelfs/zhangsiyu/Diff/DiT && \
accelerate launch --multi_gpu --num_processes 4 accelerator_train.py
"

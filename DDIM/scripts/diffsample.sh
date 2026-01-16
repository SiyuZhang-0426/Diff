#!/bin/bash
#SBATCH -p $vp
#SBATCH --gres=gpu:1

srun -p $vp --gres=gpu:1 apptainer exec --nv ~/ubuntu.sif bash -c "
source ~/.bashrc && \
conda activate diff && \
cd /mnt/petrelfs/zhangsiyu/Diff/DDIM && \
python sample.py
"

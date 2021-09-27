#!/bin/bash

#SBATCH -p palamut-cuda 
#SBATCH -A ikocdemir
#SBATCH -c 16
#SBATCH -n 1
#SBATCH -J prepare                                               # job name
#SBATCH --workdir=/truba/home/ikocdemir/mmdetection/tools/misc                      # working directory
#SBATCH --gres=gpu:1                                               # or you can specify gpu type: --gres=gpu:rtx2080ti:1
#SBATCH --output=/truba/home/ikocdemir/mmdetection/slurm_out/slurm-%j.out	        # output file
#SBATCH --error=/truba/home/ikocdemir/mmdetection/slurm_out/slurm-%j.err	                    # error file
#SBATCH --time=0-0:50:00				                            # max job time. (currently no need to use)
#SBATCH --no-requeue 

source ~/.bashrc
conda activate romer-gpu

python -u bbox_analysis_OOD.py /truba/home/ikocdemir/data/HDR4RTT/0_RESIZED/images/


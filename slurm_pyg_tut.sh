#!/bin/bash
# Job name:
#SBATCH --job-name=pyg-tut
#
# Partition:
#SBATCH --partition=es1
#
# QoS:
#SBATCH --qos=es_normal
#
# Account:
#SBATCH --account=pc_exagraph
#
# GPUS:
#SBATCH --gres=gpu:1
#
# CPU cores:
#SBATCH --cpus-per-task=2
#
# Wall clock limit:
#SBATCH --time=00:10:00

## Run command
module load python/3.6
module load ml/torch/torch7
module unload gcc/4.8.5
module load gcc/6.3.0
module unload cuda/8.0
module load cuda/10.0
source virtual3/bin/activate

nvidia-smi

# python tut-recsys.py
python pytorch_geometric/examples/gcn.py

#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=128G                                      ## Memory per node
#SBATCH --time=48:00:00                                 ## Walltime
#SBATCH --job-name=rl_bridge_uq                         ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL                                ## Mail events
#SBATCH --mail-user=zm2074@princeton.edu
#SBATCH --exclude=neu[301,306]

# activate open-world venv
source /n/fs/not-fmrl/Projects/apple_project/open-world/.venv/bin/activate

# change directory
cd /n/fs/not-fmrl/Projects/apple_project/open-world

# export environment variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# add dynuq repo to PYTHONPATH so its internal imports resolve
export PYTHONPATH=/n/fs/not-fmrl/Projects/apple_project/ac_video_model_uq:$PYTHONPATH

python scripts/run_rl_finetune.py \
    --config configs/rl/rl_bridge_uq.yaml

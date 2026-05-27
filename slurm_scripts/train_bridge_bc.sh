#!/bin/bash

#SBATCH --nodes=1                                       ## Node count
#SBATCH --gres=gpu:1                                    ## Number of GPUs per node
#SBATCH --ntasks-per-node=1                             ## Number of tasks per node
#SBATCH --cpus-per-task=8                               ## CPU cores per task
#SBATCH --mem=64G                                       ## Memory per node
#SBATCH --time=24:00:00                                 ## Walltime
#SBATCH --job-name=train_bridge_bc                      ## Job Name
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

# Pre-extract frames to JPEG for fast DataLoader reads (skips if already done).
python scripts/preprocess_bridge_frames.py --split train --num_workers 8
python scripts/preprocess_bridge_frames.py --split val   --num_workers 4

python scripts/train_bridge_bc.py \
    --dataset_path /n/fs/not-fmrl/Projects/wm_alignment/cosmos-predict2/datasets/bridge \
    --output_dir outputs/bridge_bc \
    --num_epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 42 \
    --wandb_project bridge_bc

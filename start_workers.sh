#!/bin/bash
#SBATCH --job-name=train_workers
#SBATCH --account=fc_wolflab
#SBATCH --partition=savio4_htc
#SBATCH --array=0-499
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_manager_%j.out

python work.py --worker_id $SLURM_ARRAY_TASK_ID

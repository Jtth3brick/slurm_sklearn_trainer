#!/bin/bash
#SBATCH --job-name=train_workers
#SBATCH --account=fc_wolflab
#SBATCH --partition=savio4_htc
#SBATCH --array=0-999
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_worker_%a_%j.out

python -u work.py --worker_id $SLURM_ARRAY_TASK_ID

#!/bin/bash
#SBATCH --job-name=train_manager
#SBATCH --account=ac_wolflab
#SBATCH --partition=savio3
#SBATCH --time=72:00:00
#SBATCH --output=logs/manager_%j.out

# Create directories if they don't exist
mkdir -p $(pwd)/data
mkdir -p $(pwd)/logs

# Start Redis using Apptainer
apptainer exec \
    -B $(pwd)/redis.conf:/usr/local/etc/redis/redis.conf \
    -B $(pwd)/data:/data \
    docker://redis:alpine redis-server /usr/local/etc/redis/redis.conf &

echo "$(hostname):6379" >redis_connection.txt

# Wait a moment for Redis to start
sleep 10

# Set up data and worker args queue
python setup_args.py

# keep Redis alive
echo "Setup complete. Keeping redis alive"
sleep 71h

# Make one final save with a different name
apptainer exec docker://redis:alpine redis-cli -h localhost CONFIG SET dbfilename final_save.rdb
apptainer exec docker://redis:alpine redis-cli -h localhost SAVE

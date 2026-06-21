#!/bin/bash -l

#SBATCH --job-name=ffpp_100_100_100_jpg95
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00

#SBATCH --output=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_100_100_100_jpg95-%j.out
#SBATCH --error=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_100_100_100_jpg95-%j.err

set -eo pipefail

cd /scratch/users/tsenatori/deepfakeSII

mkdir -p logs_hpc

eval "$(micromamba shell hook --shell bash)"
micromamba activate deepfake

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 64G"

python -u scripts/create_ffpp_data_hpc.py \
    --workers $SLURM_CPUS_PER_TASK \
    --train_samples 100 \
    --val_samples 100 \
    --test_samples 100 \
    --output_name "faceforensics++_6classes_100_100_100_jpg95"

echo "Job finished at: $(date)"

#!/bin/bash -l

#SBATCH --job-name=ffpp_resnet18_1_1_auto
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH -G 1

#SBATCH --output=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_resnet18_1_1_auto-%j.out
#SBATCH --error=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_resnet18_1_1_auto-%j.err

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
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

export PYTHONPATH=$PWD:$PYTHONPATH

srun python -u -m src.train \
    --dataset_name ffpp \
    --task multiclass \
    --model_name resnet18 \
    --auto_batch_size \
    --num_workers 8 \
    --train_n_frames_per_video 1 \
    --val_n_frames_per_video 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --max_epochs 50 \
    --patience 7 \
    --output_dir outputs

echo "Job finished at: $(date)"

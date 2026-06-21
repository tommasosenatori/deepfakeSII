#!/bin/bash -l

#SBATCH --job-name=ffpp_all_trainings
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -G 1
#SBATCH --array=0-17%4

#SBATCH --output=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_all_trainings-%A_%a.out
#SBATCH --error=/scratch/users/tsenatori/deepfakeSII/logs_hpc/ffpp_all_trainings-%A_%a.err

set -eo pipefail

cd /scratch/users/tsenatori/deepfakeSII

mkdir -p logs_hpc

eval "$(micromamba shell hook --shell bash)"
micromamba activate deepfake

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Job started at: $(date)"
echo "Array job id: $SLURM_ARRAY_JOB_ID"
echo "Array task id: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

CONFIGS=(
"resnet18 binary 1 1 1e-4"
"resnet18 binary 10 10 1e-4"
"resnet18 binary 100 100 1e-4"
"resnet18 multiclass 1 1 1e-4"
"resnet18 multiclass 10 10 1e-4"
"resnet18 multiclass 100 100 1e-4"

"medium_cnn binary 1 1 1e-3"
"medium_cnn binary 10 10 1e-3"
"medium_cnn binary 100 100 1e-3"
"medium_cnn multiclass 1 1 1e-3"
"medium_cnn multiclass 10 10 1e-3"
"medium_cnn multiclass 100 100 1e-3"

"small_cnn binary 1 1 1e-3"
"small_cnn binary 10 10 1e-3"
"small_cnn binary 100 100 1e-3"
"small_cnn multiclass 1 1 1e-3"
"small_cnn multiclass 10 10 1e-3"
"small_cnn multiclass 100 100 1e-3"
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

read MODEL TASK TRAIN_FRAMES VAL_FRAMES LR <<< "$CONFIG"

echo "Selected config:"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Train frames per video: $TRAIN_FRAMES"
echo "Val frames per video: $VAL_FRAMES"
echo "Learning rate: $LR"

srun python -u -m src.train \
    --dataset_name ffpp \
    --task "$TASK" \
    --model_name "$MODEL" \
    --auto_batch_size \
    --num_workers 8 \
    --train_n_frames_per_video "$TRAIN_FRAMES" \
    --val_n_frames_per_video "$VAL_FRAMES" \
    --learning_rate "$LR" \
    --weight_decay 1e-4 \
    --max_epochs 300 \
    --patience 25 \
    --output_dir outputs

echo "Job finished at: $(date)"

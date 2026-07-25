#!/bin/bash
# run_experiments.sh

# SLURM parameters
#SBATCH --job-name=ts_forecast
#SBATCH --output=logs/slurm/%x_%A_%a.o
#SBATCH --error=logs/slurm/%x_%A_%a.e
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=lp_lirisnlp
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yongbo.yu@student.kuleuven.be
#SBATCH --array=0-8  # 3 datasets × 3 horizons = 9 jobs

# Load required modules
module purge
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate pmf-benchmark
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create arrays of parameters
datasets=("RTFMP" "BPI2019" "BPI2020")
horizons=(1 3 7)

# Calculate indices
dataset_idx=$((SLURM_ARRAY_TASK_ID / ${#horizons[@]}))
horizon_idx=$((SLURM_ARRAY_TASK_ID % ${#horizons[@]}))

# Get current parameters
dataset=${datasets[$dataset_idx]}
horizon=${horizons[$horizon_idx]}

# Create scratch directory with unique name per array task
JOB_SCRATCH_DIR="$VSC_SCRATCH/pmf_benchmark_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $JOB_SCRATCH_DIR

# Copy project files
rsync -av --exclude='.git' --exclude='data/raw' --exclude='data/interim' \
    $VSC_DATA/PMF_Benchmark_2025/ $JOB_SCRATCH_DIR/

cd $JOB_SCRATCH_DIR

# Set resource configurations
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

# Run all models for current dataset and horizon
python train.py \
    --dataset "$dataset" \
    --model_group "all" \
    --horizon "$horizon" \
    --config "config/base_config.yaml"

# Copy results back
rsync -av $JOB_SCRATCH_DIR/results/ $VSC_DATA/PMF_Benchmark_2025/results/
rsync -av $JOB_SCRATCH_DIR/logs/ $VSC_DATA/PMF_Benchmark_2025/logs/

# Cleanup
rm -rf $JOB_SCRATCH_DIR

echo "All jobs completed successfully!"

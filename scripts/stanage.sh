#!/bin/bash
#SBATCH --job-name=qxdm-analysis
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/qxdm-analysis-%j.out
#SBATCH --mail-user=ziming.liu@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs

export SLURM_EXPORT_ENV=ALL


module load Anaconda3/2024.02-1
source activate /mnt/parscratch/users/$USER/envs/tspec
module load cuDNN/8.9.2.26-CUDA-12.1.1
echo "module load completed successfully!"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optimized parameters for QXDM log analysis on HPC
export BLOCK_SIZE=2000000        # Large blocks for HPC with 160GB RAM
export BATCH_SIZE=1024           # Large batch size for GPU processing

# Advanced sampling configuration for large datasets
export SAMPLE_SIZE_BASE=100000   # 100K base sample (vs default 50K)
export SAMPLE_SIZE_MAX=500000    # 500K max sample (vs default 200K)
export SAMPLE_THRESHOLD=200000   # Start sampling above 200K sentences

echo "export completed successfully!"

# QXDM logs root directory (adapt to your HPC environment)
export QXDM_ROOT=/mnt/parscratch/users/$USER/QXDM_Logs

# Create checkpoint directory for large-scale processing
mkdir -p checkpoints

# 启动QXDM日志分析作业
echo "QXDM Log Analysis Job Start!"
echo "System Configuration:"
echo "  → CPUs: $SLURM_CPUS_PER_TASK"
echo "  → Memory: 160GB"
echo "  → GPU: 1x GPU"
echo "  → Block Size: $BLOCK_SIZE"
echo "  → Batch Size: $BATCH_SIZE"
echo "  → QXDM Root: $QXDM_ROOT"

srun python src/tspec_metrics_2.py \
    --wandb-project QXDM-Complexity-Analysis \
    --checkpoint-file checkpoints/qxdm_sentences.pkl \
    --embeds-file checkpoints/qxdm_embeddings.npz \
    --log-sys \
    --sys-interval 60
# 运行完毕后，输出完成消息和结果统计
echo ""
echo "=============================================="
echo "QXDM Log Analysis Job completed successfully!"
echo "=============================================="
echo "Output files generated:"
echo "  → release_metrics.csv (per-category metrics)"
echo "  → delta_metrics.csv (cross-category comparisons)"
echo "  → checkpoints/ (sentence and embedding caches)"
echo ""
echo "Check Weights & Biases dashboard for detailed analysis:"
echo "  Project: QXDM-Complexity-Analysis"
echo "=============================================="

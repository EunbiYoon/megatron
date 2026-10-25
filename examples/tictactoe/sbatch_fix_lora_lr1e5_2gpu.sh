#!/bin/bash
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --job-name=ttt-lora-1e-5-2g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-06:00:00   # ~step 50: 1h startup + 50 x ~55min (2-GPU LoRA, measured) + margin
#SBATCH --mail-user=eunbiyoon@umass.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_10
#SBATCH --output=./slurm_logs/ttt-lora-1e-5-2g-%j.out

source /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh
conda activate /scratch/workspace/eunbiyoon_umass_edu-paper/eunbiyoon_umass_edu/.conda/envs/mg

cd /scratch/workspace/eunbiyoon_umass_edu-paper/megatron
mkdir -p slurm_logs
bash examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_fix_lora_lr1e5_2gpu.sh

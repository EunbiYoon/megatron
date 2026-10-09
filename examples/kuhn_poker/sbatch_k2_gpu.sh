#!/bin/bash
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --job-name=k2b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --exclude=gpu037,gpu038
#SBATCH --mail-user=eunbiyoon@umass.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_10
#SBATCH --output=./slurm_logs/k2b-%j.out

source /modules/opt/linux-ubuntu24.04-x86_64/miniforge3/24.7.1/etc/profile.d/conda.sh
conda activate /scratch/workspace/eunbiyoon_umass_edu-paper/eunbiyoon_umass_edu/.conda/envs/mg

cd /scratch/workspace/eunbiyoon_umass_edu-paper/megatron
mkdir -p slurm_logs
bash examples/kuhn_poker/run_agentic_pipeline_kuhn_poker_selfplay_quickcheck.sh

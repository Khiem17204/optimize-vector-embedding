#!/bin/bash
#SBATCH --job-name=lotus-index-27
#SBATCH --output=lotus_index_27.out
#SBATCH --error=lotus_index_27.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ktle@umass.edu

# Load Conda and activate environment
module load conda/latest
conda activate lotus

# Optional Hugging Face flags
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_OFFLINE=1

# Logging info
echo "Running lotus index job for split 27..."
date

# Run only split 27
python3 job.py --split-id 27 --num-splits 50

echo "Completed lotus index split 27"
date

#!/bin/bash
#SBATCH --job-name=main20_array
#SBATCH --output=slurm_logs/main20_%A_%a.out
#SBATCH --error=slurm_logs/main20_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=shared

mkdir -p slurm_logs

# --- Environment setup (modify for your cluster) ---
module load anaconda3 || module load python || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate python_env1 2>/dev/null || true

# Task count from Python (uses env POTENTIAL_MODE/POTENTIAL_FILES_JSON/COMBINED_FILE/DESIRED_PAIRS_JSON/XC_SCALES_JSON)
COUNT=$(python main2_0_Cluster.py --get-count)

# Shared batch folder
if [ -z "$BATCH_FOLDER" ]; then
  export BATCH_FOLDER="analysis_folder_cluster/$(date +%Y%m%d_%H%M%S)_main20_batch"
fi
mkdir -p "$BATCH_FOLDER"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Submitting main2_0 job array of size $COUNT to $BATCH_FOLDER"
  sbatch --array=0-$(($COUNT-1)) --export=ALL,BATCH_FOLDER="$BATCH_FOLDER" "$0"
  exit 0
fi

python main2_0_Cluster.py --task-id "${SLURM_ARRAY_TASK_ID}" --batch-folder "${BATCH_FOLDER}"



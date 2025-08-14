#!/bin/bash
#SBATCH --job-name=tf_array     # Job name
#SBATCH --output=slurm_logs/tf_array_%A_%a.out  # Standard output and error log (%A = job ID, %a = task ID)
#SBATCH --error=slurm_logs/tf_array_%A_%a.err
#SBATCH --nodes=1               # Run all tasks on a single node
#SBATCH --ntasks=1              # Run a single task
#SBATCH --cpus-per-task=8       # Number of CPU cores per task
#SBATCH --mem=8G                # Job memory request
#SBATCH --time=04:00:00         # Time limit hrs:min:sec
#SBATCH --partition=shared      # Partition name (check your cluster's documentation, e.g., 'serial_requeue', 'shared')

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# --- Environment Setup (MODIFY FOR YOUR CLUSTER) ---
# module load python/3.11.5-fasrc01
# module load anaconda3/2023.09-fasrc01
module load python
source /n/home04/rfushio/.venvs/python_env1/bin/activate

# Dynamically determine the number of tasks from the Python script (reads COMBOS_* env if set)
COUNT=$(python mainUCSB_Cluster.py --get-count)

# Create a single batch folder for the whole array (only for the first task)
if [ -z "$BATCH_FOLDER" ]; then
  export BATCH_FOLDER="analysis_folder_Cluster/$(date +%Y%m%d_%H%M%S)_batch"
fi
mkdir -p "$BATCH_FOLDER"

# If this is the master submission (no SLURM_ARRAY_TASK_ID), re-submit as a proper array
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Submitting job array of size $COUNT with batch folder $BATCH_FOLDER"
  sbatch --array=0-$(($COUNT-1)) --export=ALL,BATCH_FOLDER="$BATCH_FOLDER" "$0"
  exit 0
fi

# --- Run the Python script for this array task ---
python mainUCSB_Cluster.py --task-id "${SLURM_ARRAY_TASK_ID}" --batch-folder "${BATCH_FOLDER}"

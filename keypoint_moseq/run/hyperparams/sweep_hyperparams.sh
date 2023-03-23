# Usage: bash sweep_hyperparams.sh

JOB_SCRIPT="/home/shruthi/kpms/keypoint_moseq/run/hyperparams/sweep_hyperparams_array.sh"
WORK_DIR="/home/shruthi/fit_pair/2023_03_22-01_32_11" # Path to the directory containing the nested hyperparams directory
VID_DIR="/scratch/gpfs/shruthi/pair_wt_gold/"

ARRAY_ARGS_FILE="$WORK_DIR/array_args.txt"

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Submitting $NUM_ARRAY_JOBS jobs"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$VID_DIR"
# Usage: bash fit_hyperparams.sh

JOB_SCRIPT="/home/shruthi/kpms/keypoint_moseq/run/hyperparams/fit_models_array.sh"
WORK_DIR="/home/shruthi/fit_pair/2023_04_21-20_04_07" # Path to the directory containing the nested hyperparams directory
VID_DIR="/scratch/gpfs/shruthi/pair_wt_gold/190612_110405_wt_16276625_rig2.1/"


ARRAY_ARGS_FILE="$WORK_DIR/array_args.txt"

NUM_ARRAY_JOBS="$(cat $ARRAY_ARGS_FILE | wc -l)"

echo "Submitting $NUM_ARRAY_JOBS jobs"

sbatch -a 1-"$NUM_ARRAY_JOBS" "$JOB_SCRIPT" "$ARRAY_ARGS_FILE" "$VID_DIR"
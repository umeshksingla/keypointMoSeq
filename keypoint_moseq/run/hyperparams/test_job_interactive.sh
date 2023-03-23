#!/bin/bash

# Request interaactive node - do this on the CLI
# salloc --nodes=1 --ntasks=1 --mem=4G --time=00:30:00 --gres=gpu:1

# Activate conda environment
source activate keypoint_moseq_sleap

# Test arguments
PROJECT_DIR="/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_03_07-08_38_55/sweep_alpha/5"
VID_DIR="/scratch/gpfs/shruthi/pair_wt_gold/190612_110405_wt_16276625_rig2.1"
# PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_to_test_hyperparams.py"
PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/get_test_probs.py"


# Run script
python "$PY_SCRIPT" -p "$PROJECT_DIR" -v "$VID_DIR"

#!/bin/bash

video_dir="/scratch/gpfs/shruthi/pair_wt_gold"
save_dir="/scratch/gpfs/shruthi/pair_wt_gold/fitting"

run_script="generate_hyperparams_configs.py"

source activate keypoint_moseq_sleap

python "$run_script" -v "$video_dir" -s "$save_dir"
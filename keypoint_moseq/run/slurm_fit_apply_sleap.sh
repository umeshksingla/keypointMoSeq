#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --chdir='/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq'
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/log.%j.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

PY_SCRIPT="/home/shruthi/kpms/keypoint_moseq/run/apply_kpms_to_sleap.py"
VIDEO_DIR="/scratch/gpfs/shruthi/pair_wt_gold"
PROJECT_DIR="/scratch/gpfs/shruthi/fit_pair"
NAME="2023_04_12-11_52_08"
USE_INSTANCE=0 # instance 0 is female, instance 1 is male

source activate keypoint_moseq_sleap
python $PY_SCRIPT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --model_name $NAME --use_instance $USE_INSTANCE
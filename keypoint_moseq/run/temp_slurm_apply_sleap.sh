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
VIDEO_DIR="/scratch/gpfs/shruthi/pair_wt_gold/190612_110405_wt_16276625_rig2.1"
PROJECT_DIR="/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_04_26-21_11_07"
NAME="2023_04_26-21_12_16"
USE_INSTANCE=1 # instance 0 is female, instance 1 is male

module load cudnn/cuda-11.x/8.2.0

source activate keypoint_moseq_sleap
python $PY_SCRIPT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --model_name $NAME --use_instance $USE_INSTANCE
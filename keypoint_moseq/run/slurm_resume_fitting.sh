#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --chdir='/scratch/gpfs/shruthi/keypointMoSeq'
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/resume_fit.%j.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

VIDEO_DIR="/scratch/gpfs/shruthi/pair_wt_gold/data"
echo "$VIDEO_DIR"

PROJECT_DIR="/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_04_26-21_11_07/"
echo "$PROJECT_DIR"

USE_INSTANCE=1 # 0 for female, 1 for male
echo "$USE_INSTANCE"

RESUME_FITTING=1 # 0 for False, 1 for True
echo "$RESUME_FITTING"

CHECKPOINT_PATH="/scratch/gpfs/shruthi/pair_wt_gold/fitting/2023_04_26-21_11_07/2023_04_26-21_12_16/checkpoint.p"

PY_SCRIPT_FIT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/resume_fitting_new_data.py"
# PY_SCRIPT_INFER="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/apply_kpms_to_sleap.py"

module load cudnn/cuda-11.x/8.2.0
source activate keypoint_moseq_sleap

python $PY_SCRIPT_FIT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE --resume_fitting $RESUME_FITTING --checkpoint_path $CHECKPOINT_PATH

# python $PY_SCRIPT_INFER --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE
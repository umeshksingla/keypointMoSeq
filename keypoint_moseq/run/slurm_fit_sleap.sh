#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --chdir='/scratch/gpfs/shruthi/keypointMoSeq'
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/log.%j.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

VIDEO_DIR="/scratch/gpfs/shruthi/pair_wt_gold"
echo "$VIDEO_DIR"

PROJECT_DIR="/scratch/gpfs/shruthi/pair_wt_gold/fitting/"
echo "$PROJECT_DIR"

USE_INSTANCE=1 # 0 for female, 1 for male
echo "$USE_INSTANCE"

PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/fit_kpms_to_sleap.py"

module load cudnn/cuda-11.x/8.2.0
source activate keypoint_moseq_sleap
python $PY_SCRIPT --video_dir $VIDEO_DIR --project_dir $PROJECT_DIR --use_instance $USE_INSTANCE

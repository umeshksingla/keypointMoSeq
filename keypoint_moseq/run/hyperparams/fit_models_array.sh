#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/fit_hyp.%A.%a.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/fit_to_test_hyperparams.py"
# PY_SCRIPT="/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/hyperparams/get_test_probs.py"
echo "$PY_SCRIPT"

array_args_file="$1"
video_dir="$2"

echo "$array_args_file"
echo "$video_dir"

linenum=$SLURM_ARRAY_TASK_ID
echo "$linenum"                              
project_dir=$(sed -n "$linenum p" $array_args_file)

echo "$project_dir"

module load cudnn/cuda-11.x/8.2.0

source activate keypoint_moseq_sleap
python "$PY_SCRIPT" -p "$project_dir" -v "$video_dir"
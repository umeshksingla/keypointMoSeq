#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 
#SBATCH --chdir='/scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq'
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/log.%j.log'
#SBATCH --mail-type=begin
#SBATCH --mail-type=end 

source activate keypoint_moseq_sleap
python /home/shruthi/kpms/keypoint_moseq/run/apply_kpms_to_sleap.py
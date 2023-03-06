#!/bin/bash
#SBATCH --job-name=sweep-kappa     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1 
#SBATCH --array=0-5              # job array with index values 0, 1, 2, 3, 4
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=shruthi@princeton.edu
#SBATCH --chdir='/scratch/gpfs/shruthi/keypointMoSeq'
#SBATCH --output='/scratch/gpfs/shruthi/keypointMoSeq/logs/log.%A.%a.log'



echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

source activate keypoint_moseq_sleap

python /scratch/gpfs/shruthi/keypointMoSeq/keypoint_moseq/run/sweep_hyperparams_sleap.py

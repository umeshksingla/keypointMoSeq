"""


"""

import os
import pandas as pd
import shutil
from fit_kpms_to_sleap import find_sleap_paths, load_data_from_expts, fit_keypoint_ARHMM, fit_keypoint_SLDS
from keypoint_moseq.project.io import load_config

SLURM_ARRAY_JOB_ID = int(os.environ["SLURM_ARRAY_TASK_ID"])

def copy_updated_config_to_model_dir(project_dir, model_name):
    """
    Take the current config.yml file used to run model_name in project_dir
    and copy it to the model_name subdirectory of project_dir. 
    """

    src_file = os.path.join(project_dir, 'config.yml')
    dst_file = os.path.join(project_dir, model_name, 'config.yml')
    shutil.copyfile(src_file, dst_file)

    print(f"Copied config for model {model_name} to {dst_file}. This config looks like: ")
    print(load_config(project_dir))


def main():
    # Main control flow of the experiment
    della = True
    hyperparams_csv = None # Define settings of parameters for model fitting
    if della:
        video_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/" 
        project_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/fitting"
    else:
        video_dir = r"D:\data\pair_wt_gold"
        project_dir = r"D:\data\pair_wt_gold\fitting"
    
    # Get sleap experiment folders 
    sleap_paths = find_sleap_paths(video_dir)

    # Read data from sleap experiment folders
    data, batch_info = load_data_from_expts(sleap_paths,
                                            project_dir)

    # Update hyperparameters 

    # Initialize and fit ARHMM
    _, _, name = fit_keypoint_ARHMM(project_dir, data, batch_info)



    # Fit SLDS
    fit_keypoint_SLDS(project_dir, name)

if __name__ == "__main__":
    main()

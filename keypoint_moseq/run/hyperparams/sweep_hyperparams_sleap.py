"""
This script receives as input a project_dir that contains a specific config.yml file.
It initializes a model with the parameters specified in the config.yml file.
It fits the model with data from sleap experiments that are read and initialized as
per usual while fitting any model with experiments.
In the case of the hyperparameter sweep, the experiments are split in train-test splits.
For now, there are 5 experiments that I am fitting the models on.
I will take four of these experiments to train the model on and evaluate the model on the
fifth experiment.

The script is launched on Della using a SLURM array job that iterates over the array_args.txt
file that is generated by generate_hyperparams_configs.py. The array_args.txt file contains
a list of project_dirs.
"""

import numpy as np
import keypoint_moseq as kpm
from keypoint_moseq.run.fit_kpms_to_sleap import find_sleap_paths, load_data_from_expts, run_fit_PCA, fit_keypoint_ARHMM, fit_keypoint_SLDS
import argparse
from rich.pretty import pprint

def create_cli_parser():
    """Create an argument parser for the command line interface."""
    parser = argparse.ArgumentParser(
        description="Sweep hyperparameters for ARHMM and SLDS models on SLEAP data."
    )
    parser.add_argument(
        "-p",
        "--project_dir",
        type=str,
        required=True,
        help="Path to project directory containing config.yml file.",
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        type=str,
        required=True,
        help="Path to video directory containing sleap experiments.",
    )
    parser.add_argument('--use_instance', type=int, default=1)

    return parser

def train_test_split(sleap_paths, train_ids=None, test_ids=None):
    """
    Given a list of sleap_paths, split the experiments in sleap_paths
    to train and test splits.
    For now, I'm only considering the very fixed case where I have n experiments
    in sleap_paths and I hold out one of the experiments for testing.
    I will use the remaining n-1 experiments to train the model on.
    """
    num_expts = len(sleap_paths)
    train_ids = np.arange(0, num_expts-1, dtype = int)
    test_ids = np.array([num_expts-1], dtype = int)
    train_paths = [sleap_paths[i] for i in train_ids]
    test_paths = [sleap_paths[i] for i in test_ids]

    return train_paths, test_paths


def main():
    # Main control flow of the experiment

    # Parse CL args
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))
    print()
    video_dir = args.video_dir
    project_dir = args.project_dir
    use_instance = args.use_instance

    # Get sleap experiment folders 
    sleap_paths = find_sleap_paths(video_dir)

    # Split sleap experiments into train and test splits
    train_paths, test_paths = train_test_split(sleap_paths)

    # Load train data from train experiment folders
    data, batch_info = load_data_from_expts(train_paths,
                                            project_dir,
                                            use_instance)

    # Fit PCA
    run_fit_PCA(data, project_dir)

    # Initialize and fit ARHMM
    _, _, name = fit_keypoint_ARHMM(project_dir, data, batch_info)

    # Fit SLDS
    fit_keypoint_SLDS(project_dir, name)

    # Load test data from test experiment folders
    # coordinates = kpm.load_keypoints_from_slp_list(test_paths)
    # confidences = None

    # # Load checkpoint and config
    # checkpoint = kpm.load_checkpoint(project_dir, name)
    # config = kpm.load_config(project_dir)
    
    # Evaluate model on test data
    # results = kpm.apply_model(coordinates=coordinates, confidences=confidences, 
    #                       project_dir=project_dir, **checkpoint, **config,
    #                       plot_every_n_iters=0)


if __name__ == "__main__":
    main()

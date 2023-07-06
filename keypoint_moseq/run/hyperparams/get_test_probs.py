"""
This script takes a keypoint_moseq model parameters and some
observed data and computes the log probability of the data.

It saves the model parameters and the log probability of the data 
in a .csv file for downstream analysis with pandas and seaborn. 

This .csv file is saved in the same location as the array_args.txt
file that is generated by generate_hyperparams_configs.py.

Given a (list of) project_dir that contain within them other directories
with a results.h5 file, read the results.h5 file and unload the model
parameters from the results.h5, find the model likelihood of the data
and 

"""

from jax_moseq.models import keypoint_slds
import keypoint_moseq as kpm
from keypoint_moseq.project.fit_utils import find_sleap_paths, calculate_ll
import os
import argparse
from rich.pretty import pprint
import joblib


def create_cli_parser():
    """Create an argument parser for the command line interface."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "-p",
    "--project_dir",
    type=str,
    required=True,
    help="Path to project directory containing model checkpoint file.",
    )

    parser.add_argument(
    "-v",
    "--video_dir",
    type=str,
    required=True,
    help="Path to video directory containing data for which to compute model log likelihood.",
    )

    parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    required=True,
    help="model_name",
    )

    return parser


def find_checkpoint(project_dir):
    
    """
    Recursively search within a directory for checkpoint.p files

    Parameters
    ----------
    project_dirs : list
        A list containing paths to directories containing checkpoint.p files
    """

    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file == "checkpoint.p":
                return os.path.join(root, file)

    return None


def get_test_probs(test_paths, checkpoint_path, project_dir, save_dir, use_instance):

    # Load checkpoint
    checkpoint = kpm.load_checkpoint(path=checkpoint_path)

    # Load checkpoint and config
    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(save_dir)
    coordinates = kpm.load_keypoints_from_slp_list(test_paths, use_instance)
    confidences = None

    # Evaluate model on test data
    _, model, data = kpm.apply_model(coordinates=coordinates, confidences=confidences,
                        save_dir=save_dir, **checkpoint, **config,
                        plot_every_n_iters=0, use_saved_states=False,
                        num_iters=1, pca=pca)

    # Compute log likelihoods
    log_Y_and_model, log_Y_given_model = calculate_ll(model['states'], model['params'],
                                                      model['hypparams'], model['noise_prior'], data)
    return log_Y_and_model, log_Y_given_model


if __name__ == "__main__":

    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))

    project_dir = args.project_dir
    video_dir = args.video_dir
    model_name = args.model_name

    # Load test data from test experiment folders
    test_paths = find_sleap_paths(video_dir)
    test_paths = []

    llh1, llh2 = get_test_probs(test_paths, checkpoint, project_dir, project_dir, 1)

    print(llh1, ll2)


    
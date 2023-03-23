"""
This script receives as input a video_dir, project_dir and model_name that contains a specific config.yml file.
If the model_name is not specified then it identifies the (first) model_name/checkpoint.p file within the project_dir.

It performs inference with the model on data from sleap experiments in the video_dir.

"""
import os
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
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default=None,
        help="Path to directory with project_dir containing trained model",
    )

    return parser


def find_model_name_in_project_dir(project_dir):
    """Find the model_name in the project_dir.
    """
    # Find model_name in project_dir
    model_name = None
    for name in os.listdir(project_dir):
        if os.path.isdir(os.path.join(project_dir, name)):
            if os.path.exists(os.path.join(project_dir, name, "checkpoint.p")):
                model_name = name
                break
    if model_name is None:
        raise ValueError("No models found in project_dir.")
    return model_name


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
    model_name = args.model_name

    # Get sleap experiment folders 
    test_paths = find_sleap_paths(video_dir)

    # Load test data from test experiment folders
    coordinates = kpm.load_keypoints_from_slp_list(test_paths)
    confidences = None

    if model_name is None:
        # Find model_name in project_dir
        model_name = find_model_name_in_project_dir(project_dir)
        name = model_name
        print("Doing inference with model_name: {}".format(model_name))

    # Load checkpoint and config
    checkpoint = kpm.load_checkpoint(project_dir, name)
    config = kpm.load_config(project_dir)
    
    # Evaluate model on test data
    results = kpm.apply_model(coordinates=coordinates, confidences=confidences, 
                          project_dir=project_dir, **checkpoint, **config,
                          plot_every_n_iters=0, use_saved_states=False,
                          pca=kpm.load_pca(project_dir))
                          
    # Find log likelihood of test data
    params = checkpoint["params"]
    


if __name__ == "__main__":
    main()

"""
Apply an existing kpms model to data that it was trained on to generate a results.h5 file.
1. Given an existing project directory and model folder name 
    a. reload the model checkpoint from this.
    b. reload the existing config from this folder
2. Given sleap_paths (used to fit the model) load coordinates, confidences (if any) and format the
input data.
3. Using this checkpoint, apply the model to the data. 
"""

# Imports
import jax
import os
import argparse
from rich.pretty import pprint
from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm

# Housekeeping
print(jax.devices())
print(jax.__version__)


def create_cli_parser():
    """Create a command line interface parser."""
    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str, default=r"D:\data\pair_wt_gold",
                        help='Path to directory containing sleap-tracked data.')
    
    parser.add_argument('--project_dir', type=str, default=r"D:\data\pair_wt_gold\fitting",
                        help='Path to directory where model will be saved.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of model folder within project_dir.')
    parser.add_argument('--use_instance', type=int, default=1)
    
    return parser


def find_sleap_paths(video_dir):
    """Search recursively within video_dir to find paths to sleap-tracked expts. and files."""
    
    print(f"Searching for paths within {video_dir}")
    sleap_paths = kpm.project.get_sleap_paths(video_dir)
    print(f"Found {len(sleap_paths)} expts.")
    return sleap_paths


def load_coords_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    """
    config = kpm.load_config(project_dir)
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths,
                                                   use_instance)
    print("Printing summary of data loaded.")
    for k,v in coordinates.items():
        print(f"Session {k}")
        print(f"Data: {v.shape}")
    return coordinates


def apply_kpms_model(coordinates, project_dir, name):
    """Load checkpoint, read config, apply model."""
    checkpoint = kpm.load_checkpoint(project_dir, name)
    config = kpm.load_config(project_dir)
    confidences = None
    results = kpm.apply_model(coordinates=coordinates, confidences=confidences, 
                          project_dir=project_dir, **checkpoint, **config,
                          plot_every_n_iters=0)


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
        # raise ValueError("No models found in project_dir.")
        print(f"No models found in {project_dir}.")
    return model_name



def main():
    # Parse arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))
    print()

    # Read arguments
    video_dir = args.video_dir
    project_dir = args.project_dir
    name = args.model_name
    use_instance = args.use_instance
    
    # Setup project
    sleap_paths = find_sleap_paths(video_dir)

    # Read data
    coordinates = load_coords_from_expts(sleap_paths,
                                            project_dir,
                                            use_instance)
    if name is None:
        name = find_model_name_in_project_dir(project_dir)

    # Apply model
    apply_kpms_model(coordinates, project_dir, name)

if __name__ == "__main__":
    main()

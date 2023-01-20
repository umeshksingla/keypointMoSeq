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
from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm

# Housekeeping
print(jax.devices())
print(jax.__version__)


def find_sleap_paths(video_dir):
    """Search recursively within video_dir to find paths to sleap-tracked expts. and files."""
    
    print(f"Searching for paths within {video_dir}")
    sleap_paths = kpm.project.get_sleap_paths(video_dir)
    print(f"Found {len(sleap_paths)} expts.")
    return sleap_paths


def load_coords_from_expts(sleap_paths, project_dir):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    """
    config = kpm.load_config(project_dir)
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths)
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


def main():
    # Main control flow of the experiment
    della = True
    if della:
        project_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/fitting"
        name = "2023_01_12-20_28_34"
        video_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/"
        # video_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/190612_110953_wt_18159203_rig3.1"
    else:
        video_dir = r"D:\data\pair_wt_gold"
        project_dir = r"D:\data\pair_wt_gold\fitting"
    
    # Setup project
    sleap_paths = find_sleap_paths(video_dir)

    # Read data
    coordinates = load_coords_from_expts(sleap_paths,
                                            project_dir)

    # Apply model
    apply_kpms_model(coordinates, project_dir, name)

if __name__ == "__main__":
    main()

"""
Given a results.h5 file generated  by applying a kpms SLDS model to
experiment data, generate statistics and plots visualizing the
results.

1. Median syllable duration across all syllables
2. Per syllable median duration 
3. Trajectory plots (combined + individual)
4. Crowd movies with coordinates overlaid
5. Empirical transition probability matrix
6. Syllables to song mode
7. Continuous latent timeseries overlaid with segmentation
8. Observed pose timeseries overload with segmentation
9. Estimated coordinates overlaid with segmentation

"""

import jax
import os
import argparse
from rich.pretty import pprint

print(jax.devices())
print(jax.__version__)

from jax.config import config
config.update('jax_enable_x64', True)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import keypoint_moseq as kpm
import keypoint_moseq as kpm
from keypoint_moseq.run.fit_kpms_to_sleap import find_sleap_paths



min_duration = 15 # frames
min_usage = 0.005 # proportion of frames

def flatten_coords(coords):
    frames, _, _ = coords.shape
    return coords.reshape(frames, -1)



def create_cli_parser():

    parser = argparse.ArgumentParser(description='Visualize results of kpms SLDS model')
    parser.add_argument("video_dir", type=str, help="Directory containing video files")
    parser.add_argument("project_dir", type=str, help="Directory containing project files")
    parser.add_argument("model_name", type=str, default=None, help="Name of model to visualize")
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
        # raise ValueError("No models found in project_dir.")
        print(f"No models found in {project_dir}.")
    return model_name



def load_coordinates(project_dir, video_dir):
    """Load coordinates from sleap_paths.
    
    Parameters
    ----------
    project_dir : str
        Path to project directory.
    video_dir : str
        Path to directory containing video files.

    Returns
    -------
    coordinates

    """
    # Generate sleap paths and load data
    sleap_paths = find_sleap_paths(video_dir)
    config = kpm.load_config(project_dir)
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths, **config())
    return coordinates



def viz_flattened_timseries(x, node_names, win=None, win_size=500, offset=10,
                            stride=1, color_names=None, color_pal=None, m=None, figsz=None, lw=2, legend=True, leg_loc='center'):
    """
    Visualize flattened timeseries typically of flattened coordinates.

    Parameters
    ----------
    x : ndarray
        Array of flattened timeseries. [frames, nodes*2]
    node_names : list
        List of node names.
    win : ndarray
        Array of frames to plot. If None, a random sample is taken.
    win_size : int
        Size of window to plot.
    offset : int
        Vertical offset between nodes for ease of visualization.
    stride : int
        Stride between frames to plot.
    color_names : list
        List of color names to use for plotting.
    color_pal : list
        List of colors to use for plotting.
    m : str
        Marker style.
    figsz : tuple
        Figure size.
    lw : int
        Line width.
    legend : bool
        Whether to plot legend.
    leg_loc : str
        Legend location.

    Returns
    -------
    ax : matplotlib axis
        Current axis object.
    win: ndarray
        Array of frames plotted.
    """

    # Set colors
    if color_names is None and color_pal is None:
        color_names = ["windows blue",
                       "red",
                       "amber",
                       "faded green",
                       "dusty purple",
                       "orange",
                       "clay",
                       "pink",
                       "greyish",
                       "fuchsia",
                       "blue green",
                       "dark blue",
                       "forest green",
                       "pastel purple",
                       "salmon",
                       "dark brown"]
        colors = sns.xkcd_palette(color_names)
    elif color_pal is not None:
        colors = color_pal
    elif color_names is not None:
        colors = sns.xkcd_palette(color_names)

    if win is None:
        # Generate a sample frame at random
        frame = np.random.randint(0, x.shape[0]-win_size)
        # From this sample frame generate a timeseries
        win = np.arange(frame,frame+win_size,stride).astype('int')

   # Initialize canvas
    if figsz is not None:
        plt.figure(figsize=figsz)
    else:
        plt.figure(figsize=(20,6))

    # Plot timeseries
    for idx in range(len(node_names)):
        if m is None:
            plt.plot(win, idx*offset + x[win,idx],'-',c=colors[idx], alpha=1, label=node_names[idx], lw=lw)
        else:
            plt.plot(win, idx*offset + x[win,idx],m,c=colors[idx], alpha=1, label=node_names[idx], lw=lw)

    # Set plot parameters
    xticks_set = np.arange(win[0], win[-1], 50)
    plt.xticks(ticks=xticks_set, fontsize=14, rotation=30)
    plt.xlabel('Frames')
    if legend:
        plt.legend(loc=leg_loc)
    sns.despine()

    return plt.gca(), win



def get_bodyparts_use_bodyparts(project_dir):
    """
    Get bodyparts and use_bodyparts from config in
    project_dir.
    
    Parameters
    ----------
    project_dir : str
        Path to project directory.

    Returns
    -------
    bodyparts : list
        List of all bodyparts in sleap skeleton.
    use_bodyparts : list
        List of bodyparts to used by kpms.
    match_real_coordinates_to_estimate : list
        List of indices to match real coordinates to estimated coordinates.
    """
    config = kpm.load_config(project_dir)
    use_bodyparts = config['use_bodyparts']
    all_bodyparts = config['bodyparts']

    match_real_coordinates_to_estimate = []
    for bp in use_bodyparts:
        match_real_coordinates_to_estimate.append(all_bodyparts.index(bp))
    
    return all_bodyparts, use_bodyparts, match_real_coordinates_to_estimate


def compare_estimated_and_groundtruth_coords(project_dir, 
                                             model_name,
                                             video_dir,):
    
    """
    Make plots comparing groundtruth and estimated coordinates.
    
    Parameters
    ----------
    project_dir : str
        Path to project directory.
    model_name : str
        Name of model to visualize.
    video_dir : str
        Search for sleap videos here.
    """
    results_path = os.path.join(project_dir, model_name, "results.h5")
    coordinates = load_coordinates(project_dir, video_dir)
    results = kpm.load_results(project_dir, model_name, path=results_path)

    expt_keys_in_results = list(results.keys())
    key = expt_keys_in_results[0]

    estimated_coordinates = {k:v['estimated_coordinates'] for k,v in results.items()}

    real_coordinates = coordinates[key]
    print(real_coordinates.shape)
    estimated_coordinates = estimated_coordinates[key]
    print(estimated_coordinates.shape)

    # Match real coordinates to estimated coordinates
    _, _, match_real_coordinates_to_estimate = get_bodyparts_use_bodyparts(project_dir)
    real_coordinates = real_coordinates[:, match_real_coordinates_to_estimate, :]

    # Plot real and estimated coordinates
    


def generate_plots(project_dir, model_name, video_dir):
    """
    Generate statistics and plots visualizing the
    results.

    1. Median syllable duration across all syllables
    2. Per syllable median duration 
    3. Trajectory plots (combined + individual)
    4. Crowd movies with coordinates overlaid
    5. Empirical transition probability matrix
    6. Syllables to song mode
    7. Continuous latent timeseries overlaid with segmentation
    8. Observed pose timeseries overload with segmentation
    9. Estimated coordinates overlaid with segmentation 

    Parameters
    ----------
    project_dir : str
        Path to project directory.
    model_name : str
        Name of model to visualize.
    video_dir : str
        Path to directory containing video files.

    """
    results_path = os.path.join(project_dir, model_name, "results.h5")
    coordinates = load_coordinates(project_dir, video_dir)
    kpm.update_config(project_dir, video_dir='/tigress/MMURTHY/junyu/data/pair')
    config = kpm.load_config(project_dir)
    
    kpm.generate_trajectory_plots(coordinates,
                              name=model_name,
                              project_dir=project_dir,
                              **config,
                              min_usage=0.005,
                              min_duration=10)
    

    kpm.generate_crowd_movies(name=model_name, 
                          project_dir=project_dir, 
                          results_path=results_path, 
                          **config, 
                          sleap=True,
                         min_duration=5,
                         min_usage=0.005,)
    
    



def main():
    # Parse CL args
    parser = create_cli_parser()
    args = parser.parse_args()
    print("Args:")
    pprint(vars(args))
    print()
    video_dir = args.video_dir
    project_dir = args.project_dir
    model_name = args.model_name

    if model_name is None:
        # Find model_name in project_dir
        model_name = find_model_name_in_project_dir(project_dir)
        name = model_name




"""
Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.
"""

# Imports
import argparse
from rich.pretty import pprint
import jax
from jax.config import config
config.update('jax_enable_x64', True)
import keypoint_moseq as kpm
from pathlib import Path
from datetime import datetime
import os

# Housekeeping
print(jax.devices())
print(jax.__version__)

HYPER = {"nlags": 3,
         "kappa": 1e12,
         "alpha": 100,
         "latent_dimension": 16,
         "gamma": 1e3}


use_bodyparts = ['thorax', 'abdomen', 'wingL',
                 'wingR', 'forelegL4', 'forelegR4',
                 'midlegL4', 'midlegR4', 'hindlegL4',
                 'hindlegR4']


def create_cli_parser():
    """Create a command line interface parser."""
    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str, default=r"D:\data\pair_wt_gold",
                        help='Path to directory containing sleap-tracked data.')
    
    parser.add_argument('--project_dir', type=str, default=r"D:\data\pair_wt_gold\fitting",
                        help='Path to directory where model will be saved.')
    
    parser.add_argument('--use_instance', type=int, default=1)
    
    return parser


def find_sleap_paths(video_dir):
    """Search recursively within video_dir to find paths to sleap-tracked expts. and files."""
    
    print(f"Searching for paths within {video_dir}")
    sleap_paths = kpm.project.get_sleap_paths(video_dir)
    print(f"Found {len(sleap_paths)} expts.")
    return sleap_paths

def create_sleap_project(project_dir, sample_sleap_path, use_bodyparts=None, 
                        anterior_bodypart=['thorax'], posterior_bodypart=['abdomen']):
    """Load skeleton and tracking metadata from sample_sleap_path and initialize project_dir with a config file."""
    
    print(f"Reading sleap metadata from expt {sample_sleap_path}")
    kpm.setup_project_from_slp(project_dir, sample_slp_file=sample_sleap_path, overwrite=True)

    config = lambda: kpm.load_config(project_dir)

    if use_bodyparts is None:
        use_bodyparts = ['thorax', 'abdomen', 'wingL', 'wingR', 
                        'forelegL4', 'forelegR4', 'midlegL4', 'midlegR4', 
                        'hindlegL4', 'hindlegR4']
    kpm.update_config(project_dir,
    use_bodyparts=use_bodyparts,
    anterior_bodyparts=['thorax'], posterior_bodyparts=['abdomen'])

    print(f"Initialized {project_dir} with config.")
    print(config())

def load_data_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    Format data for modelling and move to GPU.
    """
    config = kpm.load_config(project_dir)
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths, use_instance)
    print("Printing summary of data loaded.")
    for k,v in coordinates.items():
        print(f"Session {k}")
        print(f"Data: {v.shape}")

    data, batch_info = kpm.format_data(coordinates, **config)
    return data, batch_info

def run_fit_PCA(data, project_dir):
    """
    Fit PCA to data using parameters defined in config which is read from project_dir.
    Save PCA to project_dir.
    Run scripts to plot and visualize results of PCA fitting. (TODO: How does this work when run from the CL?)
    """
    config = kpm.load_config(project_dir)
    pca = kpm.fit_pca(**data, **config, conf=None)
    kpm.save_pca(pca, project_dir)
    kpm.print_dims_to_explain_variance(pca, 0.9)
    # Visualization might cause CL calls to crash 
    # so comment and run these lines locally
    # kpm.plot_scree(pca, project_dir=project_dir)
    # kpm.plot_pcs(pca, project_dir=project_dir, **config)

def fit_keypoint_ARHMM(project_dir, data, batch_info, name=None):
    """
    Load PCA from project_dir and initialize the model.
    Fit AR-HMM
    """
    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(project_dir)
    model = kpm.initialize_model(pca=pca, **data, **config)
    # TODO: WAF to visualize initialized parameters

    model, history, name = kpm.fit_model(model, data, batch_info, ar_only=True, 
                                   num_iters=50, project_dir=project_dir, 
                                   plot_every_n_iters=0, name=name)
    # TODO: WAF to visualize AR-HMM fit parameters

    return model, history, name

def fit_keypoint_SLDS(project_dir, name):
    """
    Restore checkpoint of model in project_dir/name.
    Resume fitting with SLDS.
    """
    checkpoint = kpm.load_checkpoint(project_dir=project_dir, name=name)
    model, history, name = kpm.resume_fitting(**checkpoint, project_dir=project_dir, 
                                        ar_only=False, num_iters=200, conf=None,
                                        plot_every_n_iters=0)

def create_folder_for_project(project_dir, sample_slp_path, use_instance, hyper):
    """
    Given a base directory where results will be saved, 
    create a folder for each hyperparameter sweep.

    Parameters
    ----------
    project_dir : str
        Path to base directory where results will be saved.
    sample_slp_path : str
        Path to sleap-tracked expt. to be used to initialize config.yml
    use_instance : int
        Instance of keypoints to be used for modelling. O for female, 1 for male
    hyper : dict
        Dictionary of hyperparameters to be used for this run.
    """

    # Create base folder to save results
    if project_dir is None:
        project_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/fitting/"
    base_date_string = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    save_results_to = os.path.join(project_dir, base_date_string)
    print(f"Saving project to: {save_results_to}")

    # Get expt. paths to be used to generate base config 
    print(f"Sample sleap path: {sample_slp_path}")

    # Setup text file to log array args for slurm run
    Path(save_results_to).mkdir(parents=True, exist_ok=True)

    # Setup project directory with custom configs
    update_dict = {'use_instance': use_instance}
    for k, v in hyper.items(): # Iterate over hyperparam keys
        kpm.setup_project_from_slp(save_results_to,
                                sample_slp_file=sample_slp_path,
                                overwrite=True) # Make sure a sample slp file exists
        if k == 'kappa':
            update_dict.update({k: float(v)}) # YAML writing doesn't support np.float64 or np.int64 datatypes
        if k == 'latent_dimension':
            update_dict.update({k: int(v)})
        if k == 'alpha':
            update_dict.update({k: float(v)})
        if k == 'nlags':
            update_dict.update({k: int(v)})
    
    print("Updating config with hyperparameters: ")
    pprint(update_dict)
    kpm.update_config(save_results_to, 
                        use_bodyparts=use_bodyparts,
                        anterior_bodyparts=['thorax'],
                        posterior_bodyparts=['abdomen'],
                        slope= -0.47,
                        intercept= 0.236721,
                        PCA_fitting_num_frames = 270000,
                        keypoint_colormap = 'cool',
                        **update_dict)
    
    config = kpm.load_config(save_results_to)
    print("Config for this run: ")
    pprint(config)
    
    # Print base folder and array_args file name
    print(f"Saving results to: {save_results_to}")
    return save_results_to


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
    use_instance = args.use_instance

    # TODO: pass hyperparameters as a dictionary from CL
    hyper = HYPER
    print(f"Hyperparameters for this run: ")
    pprint(hyper)

    # Setup project
    sleap_paths = find_sleap_paths(video_dir)
    sample_sleap_path = sleap_paths[0]
    save_results_to = create_folder_for_project(project_dir, 
                                                sample_sleap_path,
                                                use_instance, 
                                                hyper)

    # Read data
    data, batch_info = load_data_from_expts(sleap_paths,
                                            save_results_to,
                                            use_instance)

    # Fit PCA
    run_fit_PCA(data, save_results_to)

    # Initialize and fit ARHMM
    _, _, name = fit_keypoint_ARHMM(save_results_to, data, batch_info)


    # Fit SLDS
    fit_keypoint_SLDS(save_results_to, name)

if __name__ == "__main__":
    main()

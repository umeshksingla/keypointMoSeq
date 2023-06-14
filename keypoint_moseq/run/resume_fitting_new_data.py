"""
Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.
"""

# Imports
import argparse
from rich.pretty import pprint

import keypoint_moseq as kpm
from keypoint_moseq.run.fit_kpms_to_sleap import find_sleap_paths, load_data_from_expts
from jax_moseq.models.keypoint_slds import model_likelihood
import os

import jax
from jax.config import config
config.update('jax_enable_x64', True) # Switch to single-precision to handle memory issues


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

expt_batch_length = 5 # Number of experimental sessions to fit at once


def create_cli_parser():
    """Create a command line interface parser."""
    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str, default=r"D:\data\pair_wt_gold",
                        help='Path to directory containing sleap-tracked data.')
    
    parser.add_argument('--project_dir', type=str, default=r"D:\data\pair_wt_gold\fitting",
                        help='Path to directory where model will be saved.')
    
    parser.add_argument('--use_instance', type=int, default=1)

    parser.add_argument('--resume_fitting', type=bool, default=False,
                        help='Whether to resume fitting from a previous model checkpoint. If True, must specify --checkpoint_path.')
    
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint to resume fitting from.')
    
    return parser


def resume_fitting_to_new_data(checkpoint_path,
                               project_dir,
                               sleap_paths):
    """
    Load checkpoint, initialize model and resume fitting to new data.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint to resume fitting from.
    project_dir : str
        Path to directory where model will be saved.
    sleap_paths : list
        List of paths to sleap-tracked experimental sessions to fit model to.
    """

    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(project_dir)
    use_instance = config["use_instance"]

    # Hack here to resume fitting when jobs crashed at i = 20
    # sleap_paths = sleap_paths[20:]
    # print(f"Hack here to resume fitting from batch starting at idx = 20. \n")
    # print(f"The paths: {sleap_paths} \n")

    log_Y_and_model = []
    log_Y_given_model = []

    # Split sleap_paths into batches of length expt_batch_length
    for i in range(0, len(sleap_paths), expt_batch_length):
        
        # Batch expt. paths
        sleap_paths_batch = sleap_paths[i:i+expt_batch_length]
        print(f"Fitting to batch {i} of {len(sleap_paths)//expt_batch_length}")
        data, batch_info = load_data_from_expts(sleap_paths_batch,
                                                project_dir,
                                                use_instance)
        
        # Load checkpoint
        if i == 0:
            checkpoint = kpm.load_checkpoint(path=checkpoint_path)
        else:
            checkpoint = kpm.load_checkpoint(project_dir, name)

        
        # Initialize a new model using saved parameters
        model = kpm.initialize_model(pca=pca, **data,
                                     params=checkpoint["params"],
                                     **config)
        
        # Resume fitting with new data
        model, history, name = kpm.fit_model(model, data, batch_info, ar_only=False, 
                        num_iters=100, project_dir=project_dir, 
                        plot_every_n_iters=0,)

        # Compute log likelihoods
        data = model['data']
        states = model['states']
        params = model['params']
        hypparams = model['hypparams']
        noise_prior = model['noise_prior']
        ll = model_likelihood(data, states, params, hypparams, noise_prior)

        log_Y_and_model.append(sum([v.item() for v in ll.values()]))
        log_Y_given_model.append(ll['Y'].item())




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
    resume_fitting = args.resume_fitting
    checkpoint_path = args.checkpoint_path

    # Load config from project_dir and print
    config = kpm.load_config(project_dir)
    pprint(config)

    # Find expt. paths to fit model to
    sleap_paths = find_sleap_paths(video_dir)

    # Resume fitting?
    if resume_fitting:
        assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist!"
        print(f"Resuming fitting from checkpoint: {checkpoint_path}")
        assert os.path.exists(project_dir), f"Project directory {project_dir} does not exist!"
        print(f"Using project directory: {project_dir}")
        
    # Fitting
    if resume_fitting:
        resume_fitting_to_new_data(checkpoint_path,
                                   project_dir,
                                   sleap_paths)


if __name__ == "__main__":
    main()

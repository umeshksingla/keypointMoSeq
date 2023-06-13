# Imports
import argparse
from rich.pretty import pprint
import numpy as np
import keypoint_moseq as kpm
from pathlib import Path
from datetime import datetime
import os
from scipy.optimize import linear_sum_assignment, minimize

def compute_state_overlap(z1, z2, K1=None, K2=None):
    """Compute overlap between two sets of discrete states."""

    assert z1.dtype == int and z2.dtype == int
    assert z1.shape == z2.shape
    assert z1.min() >= 0 and z2.min() >= 0

    K1 = z1.max() + 1 if K1 is None else K1
    K2 = z2.max() + 1 if K2 is None else K2

    overlap = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            overlap[k1, k2] = np.sum((z1 == k1) & (z2 == k2))
    return overlap


def find_permutation(z1, z2, K1=None, K2=None):
    """Find permutation of discrete states that maximizes overlap."""

    overlap = compute_state_overlap(z1, z2, K1=K1, K2=K2)
    K1, K2 = overlap.shape

    tmp, perm = linear_sum_assignment(-overlap)
    assert np.all(tmp == np.arange(K1)), "All indices should have been matched!"

    # Pad permutation if K1 < K2
    if K1 < K2:
        unused = np.array(list(set(np.arange(K2)) - set(perm)))
        perm = np.concatenate((perm, unused))

    return perm


def create_cli_parser():
    """Create a command line interface parser."""

    parser = argparse.ArgumentParser(description='Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.')
    
    parser.add_argument('--video_dir', type=str,
                        help='Path to sleap tracked data')
    
    parser.add_argument('--project_dir', type=str,
                        help='Path to directory where model and config are daved')
    
    parser.add_argument('--model_name', type=str,
                        help='Folder name for model')
    
    parser.add_argument('--time_steps', type=int, default=60,
                        help='Number of time steps to simulate')
    
    return parser


def make_results_checkpoint_path(project_dir, name):
    """Make paths to a results file and checkpoint file."""
    results_path = os.path.join(project_dir, name, 'results.h5')
    checkpoint_path = os.path.join(project_dir, name, 'checkpoint.p')
    print(results_path)
    print(os.path.isfile(results_path))

    print(checkpoint_path)
    print(os.path.isfile(checkpoint_path))

    return results_path, checkpoint_path



def get_model_paramaters_and_data_fitted(checkpoint_path):
    """Load model parameters from a checkpoint file."""

    checkpoint = kpm.load_checkpoint(checkpoint_path)
    params = checkpoint['params']

    Ab = params['Ab']
    Cd = params['Cd']
    Q = params['Q']

    A = Ab[:, :, :-1]
    nlags = Ab.shape[2]//Ab.shape[1]
    k, d = Ab.shape
    b = Ab[:, :, -1]
    C = Cd[:, :-1]
    d = Cd[:, -1]

    Y = checkpoint['Y']
    K, D = Y.shape[-2:]

    return A, b, C, d, Q, nlags, K, D


def load_results_map_syllables_and_perm(results_path):
    """Load results and map reindexed syllables to original syllables."""

    results = kpm.load_results(results_path)
    key = list(results.keys())[0]
    reindexed = results[key]['syllables_reindexed']
    original = results[key]['syllables']
    permutation = find_permutation(reindexed, original)
    return permutation



def initialize_simulations(A, b, C, d, Q, nlags, K, D, T):
    """Initialize simulations."""

    # Initialize latent states
    z = np.zeros((ntrials, T), dtype=int)
    z[:, 0] = np.random.choice(K, size=ntrials)

    # Initialize observations
    Y = np.zeros((ntrials, T, D))
    Y[:, 0] = C[z[:, 0]] @ np.random.randn(D) + d[z[:, 0]]

    # Initialize dynamics
    x = np.zeros((ntrials, T, nlags))
    x[:, 0] = np.random.randn(ntrials, nlags)

    return z, Y, x


def main():
    # Parse CL arguments
    parser = create_cli_parser()
    args = parser.parse_args()
    video_dir = args.video_dir
    project_dir = args.project_dir
    model_name = args.model_name

    # Make paths to results and checkpoint files
    results_path, checkpoint_path = make_results_checkpoint_path(project_dir, 
                                                                 model_name)

    # Load model parameters
    A, b, C, d, Q, nlags, K, D = get_model_paramaters_and_data_fitted(checkpoint_path)

    # Load results and map reindexed syllables to original syllables
    permutation = load_results_map_syllables_and_perm(results_path)

    # 


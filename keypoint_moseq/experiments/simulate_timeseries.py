"""
Build an "ideal" timeseries where the dynamics are generated similar to how we assume behavioral motifs are generated.

Groundtruth model parameters:
    - ndim: dimensionality of the timeseries (analogous to the number of keypoint (x, y) - timeseries)
    - T: number of timesteps or frames in the experiment
    - switch_every: number of frames to stochastically switch between syllables
        - mean
        - std
    - K: number of behavioral syllables
    
What is required in a minimal config to fit the model?
    
"""

def transitions(K):

    return trans_mat



def simulate_timeseries(K, ndim, T, switch_every):


    return timeseries


def write_timeseries_to_h5(timeseries, file_path, file_name):

    return


def generate_config():

    return config


def setup_config_for_simulations(project_dir, ndim, overwrite=False, **options):
    sample_config_dict = {}
    sample_config_dict['bodyparts'] = [f'bp{d}' for d in range(ndim)]
    sample_config_dict['use_bodyparts'] = sample_config_dict['bodyparts']
    sample_config_dict['skeleton'] = [' ']

    options = {**sample_config_dict, **options}
    
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    
    generate_config(project_dir, **options)

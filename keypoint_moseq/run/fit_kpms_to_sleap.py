"""
Fit a keypoint-SLDS model to sleap-tracked data from wt_gold.
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

def load_data_from_expts(sleap_paths, project_dir):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    Format data for modelling and move to GPU.
    """
    config = kpm.load_config(project_dir)
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths)
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

def fit_keypoint_ARHMM(project_dir, data, batch_info):
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
                                   plot_every_n_iters=0)
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


def main():
    # Main control flow of the experiment
    della = False
    if della:
        video_dir = r"/scratch/gpfs/shruthi/pair_wt_gold/" 
        project_dir = r"scratch/gpfs/shruthi/pair_wt_gold/fitting"
    else:
        video_dir = r"D:\data\pair_wt_gold"
        project_dir = r"D:\data\pair_wt_gold\fitting"
    
    # Setup project
    sleap_paths = find_sleap_paths(video_dir)
    sample_sleap_path = sleap_paths[0]
    create_sleap_project(project_dir, sleap_paths[0])

    # Read data
    data, batch_info = load_data_from_expts(sleap_paths,
                                            project_dir)

    # Fit PCA
    run_fit_PCA(data, project_dir)

    # Initialize and fit ARHMM
    _, _, name = fit_keypoint_ARHMM(project_dir, data, batch_info)

    # Fit SLDS
    fit_keypoint_SLDS(project_dir, name)

if __name__ == "__main__":
    main()
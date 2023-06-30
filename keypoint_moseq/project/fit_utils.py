"""

"""
import os
import numpy as np
from textwrap import fill
from datetime import datetime
import tqdm
import keypoint_moseq as kpm
from keypoint_moseq.project.fitting import resample_model, update_history
from keypoint_moseq.project.viz import plot_progress
from keypoint_moseq.project.io import save_checkpoint
from keypoint_moseq.run.constants import EXPT_BATCH_LEN
from jax_moseq.models.keypoint_slds import model_likelihood
from scipy.stats import multivariate_normal


def find_sleap_paths(video_dir):
    """Search recursively within video_dir to find paths to sleap-tracked expts. and files."""

    print(f"Searching for paths within {video_dir}")
    sleap_paths = kpm.project.get_sleap_paths(video_dir)
    print(f"Found {len(sleap_paths)} expts.")
    return sleap_paths


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


def load_coords_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    """
    coordinates = kpm.load_keypoints_from_slp_list(sleap_paths, use_instance)
    print("Printing summary of data loaded.")
    for k,v in coordinates.items():
        print(f"Session {k}")
        print(f"Data: {v.shape}")
    return coordinates


def load_data_from_expts(sleap_paths, project_dir, use_instance):
    """
    Load keypoint tracked data from sleap_paths using config info from project_dir.
    Format data for modelling and move to GPU.
    """
    config = kpm.load_config(project_dir)
    coordinates = load_coords_from_expts(sleap_paths, project_dir, use_instance)
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


def fit_mvn(data_paths, project_dir, use_instance):
    """
    Multivariate gaussian model for the pose prediction given the current pose coordinates
    """

    data, batch_info = load_data_from_expts(data_paths, project_dir, use_instance)

    n_features = data['Y'].shape[-2]
    d = data['Y'].shape[-1]
    x = data['Y'].reshape((-1, n_features*d))
    m = data['mask'].reshape((-1))
    x = x[m > 0]

    log_Y_given_mvn = 0.0
    # TODO (US): parallelize the loop
    for i in range(1, len(x)):
        p = multivariate_normal(mean=x[i - 1], cov=np.eye(n_features*d)).pdf(x[i])
        p = np.maximum(p, 0.00001)
        log_Y_given_mvn += np.log(p)
    print("log_Y_given_mvn", log_Y_given_mvn)
    return log_Y_given_mvn, len(x)


def fit_keypoint_ARHMM(project_dir, data, batch_info, name=None, return_checkpoint_path=False):
    """
    Load PCA from project_dir and initialize the model.
    Fit AR-HMM
    """
    config = kpm.load_config(project_dir)
    pca = kpm.load_pca(project_dir)
    model = kpm.initialize_model(pca=pca, **data, **config)
    # TODO: WAF to visualize initialized parameters

    model, history, name, _ = fit_model(model, data, batch_info, ar_only=True,
                                   num_iters=50, project_dir=project_dir,
                                   plot_every_n_iters=0, name=name)
    # TODO: WAF to visualize AR-HMM fit parameters
    return model, history, name


# def fit_keypoint_SLDS(project_dir, name):
#     """
#     Restore checkpoint of model in project_dir/name.
#     Resume fitting with SLDS.
#     """
#     checkpoint = kpm.load_checkpoint(project_dir=project_dir, name=name)
#     model, history, name = kpm.resume_fitting(**checkpoint, project_dir=project_dir,
#                                         ar_only=False, num_iters=200, conf=None,
#                                         plot_every_n_iters=0)


def fit_model(model,
              data,
              batch_info,
              start_iter=0,
              history=None,
              verbose=True,
              num_iters=50,
              ar_only=False,
              name=None,
              project_dir=None,
              save_data=True,
              save_states=True,
              save_history=True,
              save_every_n_iters=10,
              history_every_n_iters=10,
              states_in_history=True,
              plot_every_n_iters=10,
              save_progress_figs=True,
              calc_ll_every_n_iters=5,
              **kwargs):
    llh = {
        'log_Y_and_model': [],
        'log_Y_given_model': []
    }

    if save_every_n_iters > 0 or save_progress_figs:
        assert project_dir, fill(
            'To save checkpoints or progress plots during fitting, provide '
            'a ``project_dir``. Otherwise set ``save_every_n_iters=0`` and '
            '``save_progress_figs=False``')
        if name is None:
            name = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
            if ar_only:
                name += '_arhmm'
        savedir = os.path.join(project_dir, name)
        if not os.path.exists(savedir): os.makedirs(savedir)
        print(fill(f'Outputs will be saved to {savedir}'))

    if history is None: history = {}

    for iteration in tqdm.trange(start_iter, num_iters + 1):
        try:
            model = resample_model(data, **model, ar_only=ar_only)

            if (calc_ll_every_n_iters > 0 and (iteration % calc_ll_every_n_iters) == 0) or (iteration == num_iters):
                log_Y_and_model, log_Y_given_model = get_ll(model, data)
                llh['log_Y_and_model'].append(log_Y_and_model)
                llh['log_Y_given_model'].append(log_Y_given_model)
        except KeyboardInterrupt:
            break

        if history_every_n_iters > 0 and (iteration % history_every_n_iters) == 0 or (iteration == num_iters):
            history = update_history(history, iteration, model,
                                     include_states=states_in_history)

        if plot_every_n_iters > 0 and (iteration % plot_every_n_iters) == 0 or (iteration == num_iters):
            plot_progress(model, data, history, iteration, name=name,
                          savefig=save_progress_figs, project_dir=project_dir)

        if save_every_n_iters > 0 and (iteration % save_every_n_iters) == 0 or (iteration == num_iters):
            save_checkpoint(model, data, history, batch_info, iteration, name=name,
                            project_dir=project_dir, save_history=save_history,
                            save_states=save_states, save_data=save_data)

    return model, history, name, llh


def resume_slds_fitting_to_new_data(checkpoint_path,
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

    llh = {'log_Y_and_model': [], 'log_Y_given_model': []}

    # Split sleap_paths into batches of length expt_batch_length
    for b in range(0, len(sleap_paths), EXPT_BATCH_LEN):

        # Batch expt. paths
        sleap_paths_batch = sleap_paths[b:b + EXPT_BATCH_LEN]
        print(f"Fitting to batch {b // EXPT_BATCH_LEN} of {len(sleap_paths) // EXPT_BATCH_LEN}")
        data, batch_info = load_data_from_expts(sleap_paths_batch,
                                                project_dir,
                                                use_instance)

        # Load checkpoint
        if b == 0:
            checkpoint = kpm.load_checkpoint(path=checkpoint_path)
        else:
            checkpoint = kpm.load_checkpoint(project_dir, name)

        # Initialize a new model using saved parameters
        model = kpm.initialize_model(pca=pca, **data,
                                     params=checkpoint["params"],
                                     **config)

        # Resume fitting with new data
        model, history, name, llh_b = fit_model(model, data, batch_info, ar_only=False,
                                             num_iters=100, project_dir=project_dir,
                                             plot_every_n_iters=0, calc_ll_every_n_iters=1)
        llh['log_Y_and_model'].extend(llh_b['log_Y_and_model'])
        llh['log_Y_given_model'].extend(llh_b['log_Y_given_model'])

    return name, llh


def get_ll(model, data):
    # Compute log likelihoods
    states = model['states']
    params = model['params']
    hypparams = model['hypparams']
    noise_prior = model['noise_prior']
    ll = model_likelihood(data, states, params, hypparams, noise_prior)
    log_Y_and_model = np.sum([v.item() for v in ll.values()])
    log_Y_given_model = ll['Y'].item()
    return log_Y_and_model, log_Y_given_model


def calculate_train_bits(train_llh):

    N = train_llh['n_samples']

    # comparing model at the last iteration for each split
    fitted_split_bits = train_llh['log_Y_given_model'][:, -1] / N - train_llh['log_Y_given_mvn'] / N
    print("fitted_cv_bits", fitted_split_bits)

    total_iters = len(train_llh['log_Y_given_model'][0])
    # comparing model at each iteration for each split
    each_split_cv_bits = (train_llh['log_Y_given_model'] / np.repeat([N], total_iters, axis=0).T) - \
                         (np.repeat([train_llh['log_Y_given_mvn']], total_iters, axis=0).T / np.repeat([N], total_iters, axis=0).T)

    print("each_split_cv_bits", each_split_cv_bits)

    return fitted_split_bits, each_split_cv_bits


def calculate_test_bits(test_llh):
    N = test_llh['n_samples']

    # comparing test data LLs for each split
    fitted_split_bits = test_llh['log_Y_given_model'][0] / N - test_llh['log_Y_given_mvn'] / N
    return fitted_split_bits


def print_ll(llh):

    print(">> Train log_Y_given_mvn:", llh['train']['log_Y_given_mvn'])

    ll = np.array(llh['train']['log_Y_and_model'])
    mean = np.mean(ll, axis=0)
    std = np.std(ll, axis=0)
    print("Train log_joint mean:", mean)
    print("Train log_joint std:", std)

    # import matplotlib.pyplot as plt
    # plt.errorbar(np.arange(ll.shape[1]), mean, yerr=std, label='Mean', fmt='-o')

    ll = np.array(llh['train']['log_Y_given_model'])
    mean = np.mean(ll, axis=0)
    std = np.std(ll, axis=0)
    print("Train log_data_ll mean:", mean)
    print("Train log_data_ll std:", std)

    print(">> Test log_Y_given_mvn:", llh['test']['log_Y_given_mvn'])

    ll = np.array(llh['test']['log_Y_and_model'])
    mean = np.mean(ll, axis=0)
    std = np.std(ll, axis=0)
    print("Test log_joint mean:", mean)
    print("Test log_joint std:", std)

    ll = np.array(llh['test']['log_Y_given_model'])
    mean = np.mean(ll, axis=0)
    std = np.std(ll, axis=0)
    print("Test log_data_ll mean:", mean)
    print("Test log_data_ll std:", std)
    return

import os
from typing import List

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.ndimage
from scipy.interpolate import interp1d

# Skeleton metadata
fly_skeleton = [
    (1, 0),
    (0, 11),
    (0, 12),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
]

fly_nodes = [
 'head',
 'thorax',
 'abdomen',
 'wingL',
 'wingR',
 'forelegL4',
 'forelegR4',
 'midlegL4',
 'midlegR4',
 'hindlegL4',
 'hindlegR4',
 'eyeL',
 'eyeR']

min_sine_wing_ang = 30

def load_tracks(expt_folder):
    """Load proofread and exported pose tracks.
    
    Args:
        expt_folder: Path to experiment folder containing inference.cleaned.proofread.tracking.h5.
    
    Returns:
        Tuple of (tracks, node_names).
        
        tracks contains the pose estimates in an array of shape (frame, joint, xy, fly).
        The last axis is ordered as [female, male].
        
        node_names contains a list of string names for the joints.
    """
    if os.path.isdir(expt_folder):
        track_file = os.path.join(expt_folder, "inference.cleaned.proofread.tracking.h5")
    else:
        track_file = expt_folder
    with h5py.File(track_file, "r") as f:
        tracks = np.transpose(f["tracks"][:])  # (frame, joint, xy, fly)
        node_names = f["node_names"][:]
        node_names = [x.decode() for x in node_names]
        
    # Crop to valid range.
    last_fidx = np.argwhere(np.isfinite(tracks.reshape(len(tracks), -1)).any(axis=-1)).squeeze()[-1]
    tracks = tracks[:last_fidx]

    return tracks, node_names


def compute_wing_angles(x, left_ind=3, right_ind=4):
    """Returns the wing angles in degrees from normalized pose.
    
    Args:
        x: Egocentric pose of shape (..., joints, 2). Use normalize_to_egocentric on the
            raw pose coordinates before passing to this function.
        left_ind: Index of the left wing. Defaults to 3.
        right_ind: Index of the right wing. Defaults to 4.
    
    Returns:
        Tuple of (thetaL, thetaR) containing the left and right wing angles.
        
        Both are in the range [-180, 180], where 0 is when the wings are exactly aligned
        to the midline (thorax to head axis).
        
        Positive angles denote extension away from the midline in the direction of the
        wing. For example, a right wing extension may have thetaR > 0.
    """
    xL, yL = x[..., left_ind, 0], x[..., left_ind, 1]
    xR, yR = x[..., right_ind, 0], x[..., right_ind, 1]
    thetaL = np.rad2deg(np.arctan2(yL, xL)) + 180
    thetaL[np.greater(thetaL, 180, where=np.isfinite(thetaL))] -= 360
    thetaR = np.rad2deg(np.arctan2(yR, xR)) + 180
    thetaR[np.greater(thetaR, 180, where=np.isfinite(thetaR))] -= 360
    thetaR = -thetaR
    
    return thetaL, thetaR




def connected_components1d(x, return_limits=False):
    """
    Return the indices of the connected components in a 1D logical array.
    
    Args:
        x: 1d logical (boolean) array.
        return_limits: If True, return indices of the limits of each component rather
            than every index. Defaults to False.
            
    Returns:
        If return_limits is False, a list of (variable size) arrays are returned, where
        each array contains the indices of each connected component.
        
        If return_limits is True, a single array of size (n, 2) is returned where the
        columns contain the indices of the starts and ends of each component.
        
    """
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends = [cc[0].stop for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def get_expt_sync(expt_folder):
    """Computes the sample/frame maps from experiment synchronization.
    
    Args:
        expt_folder: Path to experiment folder with daq.h5.
    
    Returns:
        frame_daq_sample: A vector of the length of the number of frames where each
            element is the estimated DAQ sample index.
        daq_frame_idx: A vector of the lenght of the number of samples where each
            element is the estimated video frame index.
    """
    with h5py.File(os.path.join(expt_folder, "daq.h5"), "r") as f:
        try:
            trigger = f["sync"][:]
        except KeyError:
            trigger = f["Sync"][:]

    # Threshold exposure signal.
    trigger[trigger[:] < 1.5] = 0
    trigger[trigger[:] > 1.5] = 1

    # Find connected components.
    daq2frame, n_frames = scipy.ndimage.measurements.label(trigger)
    
    # Compute sample at each frame.
    frame_idx, frame_time, count = np.unique(daq2frame, return_index=True,
                                             return_counts=True)
    frame_daq_sample = frame_time[1:] + (count[1:] - 1) / 2

    # Interpolate frame at each sample.
    f = scipy.interpolate.interp1d(
        frame_daq_sample,
        np.arange(frame_daq_sample.shape[0]),
        kind="nearest",
        fill_value="extrapolate"
    )
    daq_frame_idx = f(np.arange(trigger.shape[0]))
    
    return frame_daq_sample, daq_frame_idx


def load_song(expt_folder, return_audio=False):
    """Load song segmentation.
    
    Args:
        expt_folder: Path to experiment folder with daq_segmented.mat.
        return_audio: If True, return merged audio track. Defaults to False.
    
    Returns:
        pslow, pfast, sine: boolean vectors denoting whether song is detected at each
        sample of the recording.
        
        pulse_bout_lims, sine_bout_lims, mix_bout_lims: (n, 2) arrays containing the
        start and end sample indices for predicted bouts.
            
        If return_audio is True, then also returns a vector with the merged audio.
    """
    seg_path = os.path.join(expt_folder, "daq_segmented.mat")
    var_names = ["sine", "pfast", "pslow", "bInf"]
    if return_audio:
        var_names.append("song")
    seg = scipy.io.loadmat(seg_path, variable_names=var_names)
    
    # Bout sample limits.
    bout_lims = seg["bInf"]["stEn"][0][0]
    pulse_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Pul")[0]]
    sine_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Sin")[0]]
    mix_bouts = bout_lims[np.where(seg["bInf"]["Type"][0][0] == "Mix")[0]]

    # Masks.
    pslow = (seg["pslow"] > 0).squeeze()
    pfast = (seg["pfast"] > 0).squeeze()
    sine = (seg["sine"] > 0).squeeze()

    if return_audio:
        song = seg["song"].squeeze()
        return pslow, pfast, sine, pulse_bouts, sine_bouts, mix_bouts, song
    else:
        return pslow, pfast, sine, pulse_bouts, sine_bouts, mix_bouts



def h5read(filename, dataset):
    """Load a single dataset from HDF5 file.
    
    Args:
        filename: Path to HDF5 file.
        dataset: Name of the dataset.
    
    Returns:
        The dataset data loaded in.
    """
    with h5py.File(filename, "r") as f:
        return f[dataset][:]


def describe_hdf5(filename, attrs=True):
    """Describe all items in an HDF5 file."""
    def desc(k, v):
        if type(v) == h5py.Dataset:
            print(f"[ds]  {v.name}: {v.shape} | dtype = {v.dtype}")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")
        elif type(v) == h5py.Group:
            print(f"[grp] {v.name}:")
            if attrs and len(v.attrs) > 0:
                print(f"      attrs = {dict(v.attrs.items())}")

    with h5py.File(filename, "r") as f:
        f.visititems(desc)


def encode_hdf5_strings(S):
    """Encodes a list of strings for writing to a HDF5 file.
    
    Args:
        S: List of strings.
    
    Returns:
        List of numpy arrays that can be written to HDF5.
    """
    return [np.string_(x) for x in S]


def decode_hdf5_strings(S):
    """Decodes a list of strings stored in a HDF5 file.
    
    Args:
        S: List of encoded strings.
    
    Returns:
        List of strings.
    """
    return [x.decode() for x in S[:].tolist()]


def get_filesize(filepath):
    """Returns filesize in bytes."""
    return os.stat(filepath).st_size


def fill_missing(x, kind="nearest", **kwargs):
    """Fill missing values in a timeseries.
    
    Args:
        x: Timeseries of shape (time, _) or (_, time, _).
        kind: Type of interpolation to use. Defaults to "nearest".
    
    Returns:
        Timeseries of the same shape as the input with NaNs filled in.
    
    Notes:
        This uses pandas.DataFrame.interpolate and accepts the same kwargs.
    """
    if x.ndim == 3:
        return np.stack([fill_missing(xi, kind=kind, **kwargs) for xi in x], axis=0)
    return pd.DataFrame(x).interpolate(kind=kind, axis=0, **kwargs).to_numpy()


def fill_missing_tracks(tracks, kind="spline", **kwargs):
    """Fill missing values in tracking data.
    
    Args:
        tracks: Pose of shape (time, joints, 2)
        kind: Type of interpolation to use. Defaults to cubic splines.
        
    Returns:
        Pose timeseries that are the same shape as input tracks with NaNs filled in
        
    """
    initial_shape = tracks.shape
    tracks = tracks.reshape((initial_shape[0],-1))
    
    # Interpolate along each slice.
    for i in range(tracks.shape[-1]):
        y = tracks[:, i]

        tracks[:, i] = pd.Series(y).interpolate(kind=kind, **kwargs).to_numpy()

    #Restore to initial shape.
    tracks = tracks.reshape(initial_shape)
    return tracks

# Alternative fill missing function from Shruthi's data management pipeline - this doesn't use pandas, but uses scipy.interpolate directly
def fill_missing_tracks_SR(Y, kind="linear"):
    initial_shape = Y.shape


    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    print(f"\tY_initial.shape = {initial_shape}, Y.shape={Y.shape}")

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        non_missing_mask = ~np.isnan(y)
        num_non_missing = np.sum(non_missing_mask)

        # If we don't have enough points, don't interpolate, if we only have 2-3,
        # use linear interpolation, otherwise, use the interpolation requested.
        if num_non_missing <= 1:
            continue
        elif num_non_missing <= 4:
            kind_i = "linear"
        else:
            kind_i = kind

        # Build interpolant.
        x = np.flatnonzero(non_missing_mask)
        f = interp1d(x, y[x], kind=kind_i, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        Y[xq, i] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(Y[:, i])
        Y[:, i][mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Y[:, i][~mask])

    #Restore to initial shape.
    Y = Y.reshape(initial_shape)
    return Y


def normalize_to_egocentric(x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, fill=False, return_angles=False):
    """Normalize pose estimates to egocentric coordinates.
    
    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.
    
    Returns:
        Egocentrically aligned poses of the same shape as the input.
        
        If return_angles is True, also returns a vector of angles.
    """
        
    if rel_to is None:
        rel_to = x
    
    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)
    
    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)
    
    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    if fill:
        ctr = fill_missing(ctr, kind="nearest")
        fwd = fill_missing(fwd, kind="nearest")
    ego_fwd = fwd - ctr
    
    # Compute angle.
    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)
    
    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1
    
    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor
    
    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]
    
    if is_singleton:
        x = x[0]
    
    if return_angles:
        return x, ang
    else:
        return x


# Combine utils to read in tracking information from any given experimental folder and generate interpolated tracks and normalized to egocentric tracks

def make_expt_dataset(expt_folder, output_path=None, overwrite=False, ctr_ind=1, fwd_ind=0):
    """Gather experiment data into a single file.
    
    Args:
        expt_folder: Full absolute path to the experiment folder.
        output_path: Path to save the resulting dataset to. Can be specified as a folder
            or full path ending with ".h5". Defaults to saving to current folder. If a
            folder is specified, the dataset filename will be the experiment folder
            name with ".h5".
        overwrite: If True, overwrite even if the output path already exists. Defaults
            to False.
         ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
            
    Returns:
        Path to output dataset.
    """
    expt_name = os.path.basename(expt_folder)
    
    if output_path is None:
        output_path = os.getcwd()
    
    if not output_path.endswith(".h5"):
        output_path = os.path.join(output_path, f"{expt_name}.h5")
    
    if os.path.exists(output_path) and not overwrite:
        return output_path
    
    # Load tracking.
    tracks, node_names = load_tracks(expt_folder)
    
    # Compute tracking-related features.
    trxF = tracks[..., 0]
    trxM = tracks[..., 1]
    
    # Fill missing values
    trxF = fill_missing_tracks_SR(trxF)
    trxM = fill_missing_tracks_SR(trxM)
    
    # Normalize to egocentric - self
    egoF = normalize_to_egocentric(trxF)
    egoM = normalize_to_egocentric(trxM)
    
    # Normalize to egocentric - other
    egoFrM = normalize_to_egocentric(trxF, rel_to=trxM, ctr_ind=ctr_ind, fwd_ind=fwd_ind)
    egoMrF = normalize_to_egocentric(trxM, rel_to=trxF, ctr_ind=ctr_ind, fwd_ind=fwd_ind)

    # Compute wing angles
    wingML, wingMR = compute_wing_angles(egoM)
    wingFL, wingFR = compute_wing_angles(egoF)

    # Get sync
    sample_at_frame, frame_at_sample = get_expt_sync(expt_folder)

    # Load song
    pslow, pfast, sine, pulse_bouts, sine_bouts, mix_bouts = \
    load_song(expt_folder, return_audio=False)

    pslow_lims = connected_components1d(pslow, return_limits=True)
    pfast_lims = connected_components1d(pfast, return_limits=True)
    sine_lims = connected_components1d(sine, return_limits=True)


    # Filter out invalid song (outside of video bounds).
    s0 = sample_at_frame[0]
    s1 = sample_at_frame[len(tracks) - 1]
    pslow_lims = pslow_lims[(pslow_lims[:, 0] >= s0) & (pslow_lims[:, 1] <= s1)]
    pfast_lims = pfast_lims[(pfast_lims[:, 0] >= s0) & (pfast_lims[:, 1] <= s1)]
    sine_lims = sine_lims[(sine_lims[:, 0] >= s0) & (sine_lims[:, 1] <= s1)]
    
    pulse_bouts = pulse_bouts[(pulse_bouts[:, 0] >= s0) & (pulse_bouts[:, 1] <= s1)]
    sine_bouts = sine_bouts[(sine_bouts[:, 0] >= s0) & (sine_bouts[:, 1] <= s1)]
    mix_bouts = mix_bouts[(mix_bouts[:, 0] >= s0) & (mix_bouts[:, 1] <= s1)]
    
    # Filter out sine lims without minimum wing angle.
    valid_sine_lims = []
    for s0, s1 in sine_lims:
        f0 = int(frame_at_sample[s0])
        f1 = int(frame_at_sample[s1])
        wing_angs = np.concatenate([wingML[f0:f1], wingMR[f0:f1]])
        if (~np.isnan(wing_angs)).any() and (np.nanmax(wing_angs) > min_sine_wing_ang):
            valid_sine_lims.append(True)
        else:
            valid_sine_lims.append(False)
    valid_sine_lims = np.stack(valid_sine_lims)
    sine_lims = sine_lims[valid_sine_lims]

    print('Filtered sine lims by wing angle')

    # Filter out sine bouts without minimum wing angle.
    valid_sine_bouts = []
    for s0, s1 in sine_bouts:
        f0 = int(frame_at_sample[s0])
        f1 = int(frame_at_sample[s1])
        wing_angs = np.concatenate([wingML[f0:f1], wingMR[f0:f1]])
        if (~np.isnan(wing_angs)).any() and (np.nanmax(wing_angs) > min_sine_wing_ang):
            valid_sine_bouts.append(True)
        else:
            valid_sine_bouts.append(False)
    valid_sine_bouts = np.stack(valid_sine_bouts)
    sine_bouts = sine_bouts[valid_sine_bouts]

    print('Filtered sine bouts by wing angle')

    print('After culling sine by wing angle: ')
    for song_type, song in zip([pulse_bouts, sine_bouts, mix_bouts], ['Pul', 'Sin', 'Mix']):
        print(f'This session contains {len(song_type)} bouts of {song}')

    # Find instances of sine and pulse and mix frames
    sine_frames, pulse_frames, mix_frames = [], [], []

    for song_sample, song_frames in zip([pulse_bouts, sine_bouts, mix_bouts], [pulse_frames, sine_frames, mix_frames]):
        for s_st, s_en in song_sample:
            song_frames.append([frame_at_sample[s_st],frame_at_sample[s_en]])


    # Ensure output folder exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save.
    with h5py.File(output_path, "w") as f:
        f.create_dataset("expt_name", data=expt_name)
        f.create_dataset("expt_folder", data=expt_folder)
        f.create_dataset("node_names", data=encode_hdf5_strings(node_names))

        f.create_dataset("sample_at_frame", data=sample_at_frame, compression=1)
        f.create_dataset("frame_at_sample", data=frame_at_sample, compression=1)

        f.create_dataset("pslow_lims", data=pslow_lims, compression=1)
        f.create_dataset("pfast_lims", data=pfast_lims, compression=1)
        f.create_dataset("sine_lims", data=sine_lims, compression=1)
        f.create_dataset("pulse_bouts", data=pulse_bouts, compression=1)
        f.create_dataset("sine_bouts", data=sine_bouts, compression=1)
        f.create_dataset("mix_bouts", data=mix_bouts, compression=1)

        f.create_dataset("trxF", data=trxF, compression=1)
        f.create_dataset("trxM", data=trxM, compression=1)
        f.create_dataset("egoF", data=egoF, compression=1)
        f.create_dataset("egoM", data=egoM, compression=1)
        f.create_dataset("egoFrM", data=egoFrM, compression=1)
        f.create_dataset("egoMrF", data=egoMrF, compression=1)
        f.create_dataset("wingFL", data=wingFL, compression=1)
        f.create_dataset("wingFR", data=wingFR, compression=1)
        f.create_dataset("wingML", data=wingML, compression=1)
        f.create_dataset("wingMR", data=wingMR, compression=1)
    
    return output_path


def load_joint_ts(path_to_proc, joint_names=None, df_name="egoM"):
    """
    path_to_proc points to a *.processed.h5 file that
    contains `egoM` dataset and a `node_names` dataset
    load_joints_ts returns a 2D array [frames x len(joint_names)x2]
    and the an expanded list of joint_names appended with x or y
    for the x and y coordinates
    """
    if joint_names is None:
        joint_names = ["wingL", "wingR"]
    with h5py.File(path_to_proc, 'r') as f:
        data = np.copy(f[df_name])
        node_names = list(np.copy(f["node_names"]).astype('U13'))

    if df_name == "egoM":
        joint_idxs = []
        expand_joint_names = []

        for jt in joint_names:
            joint_idxs.append(node_names.index(jt))
            expand_joint_names.append(f"{jt}_x")
            expand_joint_names.append(f"{jt}_y")

        joint_ts = np.copy(data[:,joint_idxs,:]).reshape((data.shape[0]), -1, order='A')

        return joint_ts, expand_joint_names

    else:
        return data, node_names
    
    
    
def get_song_frames(expt_folder):
    # Get sync
    sample_at_frame, frame_at_sample = get_expt_sync(expt_folder)

    # Load song
    pslow, pfast, sine, pulse_bouts, sine_bouts, mix_bouts = \
    load_song(expt_folder, return_audio=False)

    pslow_lims = connected_components1d(pslow, return_limits=True)
    pfast_lims = connected_components1d(pfast, return_limits=True)
    sine_lims = connected_components1d(sine, return_limits=True)

    # Load tracks
    tracks, node_names = load_tracks(expt_folder)


    # Compute wing angles for male
    trxM = tracks[..., 1]
    
    # Fill missing values
    trxM = fill_missing_tracks_SR(trxM)
    
    # Normalize to egocentric - self
    egoM = normalize_to_egocentric(trxM)

    # Compute wing angles
    wingML, wingMR = compute_wing_angles(egoM)

    # Filter out invalid song (outside of video bounds).
    s0 = sample_at_frame[0]
    s1 = sample_at_frame[len(tracks) - 1]
    pslow_lims = pslow_lims[(pslow_lims[:, 0] >= s0) & (pslow_lims[:, 1] <= s1)]
    pfast_lims = pfast_lims[(pfast_lims[:, 0] >= s0) & (pfast_lims[:, 1] <= s1)]
    sine_lims = sine_lims[(sine_lims[:, 0] >= s0) & (sine_lims[:, 1] <= s1)]
    
    pulse_bouts = pulse_bouts[(pulse_bouts[:, 0] >= s0) & (pulse_bouts[:, 1] <= s1)]
    sine_bouts = sine_bouts[(sine_bouts[:, 0] >= s0) & (sine_bouts[:, 1] <= s1)]
    mix_bouts = mix_bouts[(mix_bouts[:, 0] >= s0) & (mix_bouts[:, 1] <= s1)]
    
    # Filter out sine lims without minimum wing angle.
    valid_sine_lims = []
    for s0, s1 in sine_lims:
        f0 = int(frame_at_sample[s0])
        f1 = int(frame_at_sample[s1])
        wing_angs = np.concatenate([wingML[f0:f1], wingMR[f0:f1]])
        if (~np.isnan(wing_angs)).any() and (np.nanmax(wing_angs) > min_sine_wing_ang):
            valid_sine_lims.append(True)
        else:
            valid_sine_lims.append(False)
    valid_sine_lims = np.stack(valid_sine_lims)
    sine_lims = sine_lims[valid_sine_lims]

    print('Filtered sine lims by wing angle')

    # Filter out sine bouts without minimum wing angle.
    valid_sine_bouts = []
    for s0, s1 in sine_bouts:
        f0 = int(frame_at_sample[s0])
        f1 = int(frame_at_sample[s1])
        wing_angs = np.concatenate([wingML[f0:f1], wingMR[f0:f1]])
        if (~np.isnan(wing_angs)).any() and (np.nanmax(wing_angs) > min_sine_wing_ang):
            valid_sine_bouts.append(True)
        else:
            valid_sine_bouts.append(False)
    valid_sine_bouts = np.stack(valid_sine_bouts)
    sine_bouts = sine_bouts[valid_sine_bouts]

    print('Filtered sine bouts by wing angle')

    print('After culling sine by wing angle: ')
    for song_type, song in zip([pulse_bouts, sine_bouts, mix_bouts], ['Pul', 'Sin', 'Mix']):
        print(f'This session contains {len(song_type)} bouts of {song}')

    # Find instances of sine and pulse and mix frames
    sine_frames, pulse_frames, mix_frames = [], [], []

    for song_sample, song_frames in zip([pulse_bouts, sine_bouts, mix_bouts], [pulse_frames, sine_frames, mix_frames]):
        for s_st, s_en in song_sample:
            song_frames.append([frame_at_sample[s_st],frame_at_sample[s_en]])
            
    return song_frames
import os
import shutil
from time import strftime, time

import matplotlib.pyplot as plt
import numpy as np
import rich
import seaborn as sns
import sleap

from .wtgold_utils_SR import fill_missing_tracks_SR, load_tracks

sleap.use_cpu_only()

sns.set_style("white")
sns.set_context("poster")
skel_cmap_name = "deep"
skel_cmap = sns.color_palette(skel_cmap_name)

def imgfig(img=None, figsize=(6, 6), dpi=120, **kwargs):
    """
	Take an image, create a matplotlib fig,
	plot image and return the figure handle
	"""

    fig = plt.figure(figsize=figsize, frameon=False, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    plt.imshow(img, cmap="gray", aspect="auto", interpolation="nearest", **kwargs)
    is_rgb = img.shape[-1] == 3
    if not is_rgb:
        plt.clim(0, 255)

    plt.axis("off")

    return fig


def plot_skel(pts, skeleton, full_skel=False, nodes=True, node_kws=None, edge_kws=None, **kwargs):
    """
	Take the points associated with the
	nodes of a skeleton for a particular SLEAP
	instance and plot the skeleton
	"""
    node_kws = node_kws or {}
    edge_kws = edge_kws or {}

    # flatten into sequence of: src, dst, nan
    if full_skel:
        edge_pts = np.concatenate([
            pts[np.array(skeleton.edge_inds), :],
            np.full((len(skeleton.edge_inds), 1, 2), np.nan)
        ], axis=1).reshape(-1, 2)

        plt.plot(edge_pts[:, 0], edge_pts[:, 1], "-", **edge_kws, **kwargs)


    if nodes:
        plt.plot(pts[:, 0], pts[:, 1], ".", **node_kws, **kwargs)


def square_lims(xl, yl):
    xl, yl = np.array(xl), np.array(yl)
    is_wide = xl.ptp() > yl.ptp()
    if is_wide:
        yl = np.array([-0.5, 0.5]) * xl.ptp() + yl.mean()
    else:
        xl = np.array([-0.5, 0.5]) * yl.ptp() + xl.mean()
    return xl, yl


def get_lims(pts, square=True):
    xl = np.array([np.nanmin(pts[..., 0]), np.nanmax(pts[..., 0])])
    yl = np.array([np.nanmin(pts[..., 1]), np.nanmax(pts[..., 1])])
    if square:
        xl, yl = square_lims(xl, yl)

    return xl, yl


def render_clip(expt_path,
                fidxs,
                cmap=skel_cmap_name,
                ms=4,
                lw=1.5,
                alpha=0.8,
                lims="fit",
                padding=8,
                dpi=120,
                fig_size_px=256,
                render_stride=3,
                fps=10,
                crf=20,
                save_frames=True,
                save_path=None):
    """
	Render a clip from a SLEAP tracked
	"""

    # Read in tracks
    tracks, node_names = load_tracks(expt_path)

    expt_id = os.path.split(expt_path)[-1]

    # Grab male tracks and fill missing
    trx = tracks[..., 1]
    trx = fill_missing_tracks_SR(trx, kind="linear")

    # Make movie clip name
    render_fname = f"{expt_id}_frames_{fidxs[0]}-{fidxs[-1]}.mp4"

    # Setup path to render video
    if save_path is None:
        save_path = "/tigress/MMURTHY/shruthi/movies"

    render_path = os.path.join(save_path, render_fname)

    if type(cmap) == str:
        cmap = sns.color_palette(cmap)

    print(f"Saving to {render_path}")

    # Load skeleton
    # Load sleap data and read in skeleton info
    clip = sleap.load_file(f'{expt_path}/inference.cleaned.proofread.slp')
    skeleton = clip.skeleton

    print(f"Read in skeleton")

    if type(lims) == str and lims == "fit":
        pad = np.array([-1, 1]) * padding
        xl, yl = get_lims(trx)
        xl = xl + pad
        yl = yl[::-1] + pad
    elif lims is None:
        xl, yl = None, None
    else:
        xl, yl = lims

    if save_frames:
        TMP_DIR = "../tmp/render"
        # Clear temp dir for the frames
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        os.makedirs(TMP_DIR, exist_ok=True)
        print(f"Saving images to: {TMP_DIR}")

    print(f"Num frames to render: {len(fidxs)}")
    print(f"Shape of trx: {trx.shape}")
    print(f"Num fidxs: {len(fidxs)}")

    if render_stride > 1:
        fidxs = fidxs[::render_stride]
        trx = trx[::render_stride]

    print("Updates tracks and fidxs to reflect render stride")

    render_t0 = time()
    if save_frames:
        _iterator = np.arange(len(fidxs))
        _iterator = rich.progress.track(_iterator)
    else:
        _iterator = [0]

    for t in _iterator:

        # Get frame data
        fidx = fidxs[t]
        img = clip.video[fidx].squeeze()
        poses = trx[fidx]
        #         state_label = state_labels[t]

        # Plot image and zoom
        fig = imgfig(img, figsize=(fig_size_px / dpi, fig_size_px / dpi), dpi=dpi)
        if xl is not None:
            plt.xlim(xl)
        if yl is not None:
            plt.ylim(yl)

        # Plot poses
        plot_skel(poses, skeleton, c=cmap[0], zorder=20, ms=ms, lw=lw, alpha=alpha)

        if save_frames:
            # Render frame
            plt.savefig(f"{TMP_DIR}/frame.{t:04}.png", dpi=dpi)
            plt.close(fig)

    if save_frames:
        # Render frames into video
        os.system(
            "ffmpeg -y -framerate {fps} -i '{TMP_DIR}/frame.%04d.png' -c:v libx264 -preset superfast -pix_fmt yuv420p -crf {crf} '{TMP_DIR}/movie.mp4' >nul 2>nul")
        shutil.copy(f"{TMP_DIR}/movie.mp4", render_path)
        print(f"Rendered movie: {TMP_DIR}")
        print(f"Copied to: {render_path}")

    render_dt = time() - render_t0
    print(f"Rendered in {render_dt:.1f} s")

    # Clear tmp directory
    shutil.rmtree(TMP_DIR, ignore_errors=True)


def make_egocentric_skeleton_gifs(trx,  # egocentrically aligned tracks to plot with all nodes
                              expt_path,  # path to inference.cleaned.proofread.slp to read in skeleton and video info
                              fidxs,  # frame indices
                              render_nodes, # indices to render over skeleton
                              save_dir,  # saving or working directory
                              render_fname,  # filename to save video to
                              cmap=skel_cmap_name, # cmap for skels
                              draw_skel=True,
                              draw_nodes=False,
                              ms=8,  # marker size for skel nodes
                              lw=1.5,  # linewidth for skel edges
                              alpha=0.8,  # transparency
                              lims="fit",
                              padding=8,
                              dpi=120,
                              fig_size_px=512,
                              render_stride=1,
                              fps=10,
                              crf=20,
                              save_frames=True):

    if type(cmap) == str:
        cmap = sns.color_palette(cmap)

    timestr = strftime("%Y%m%d-%H%M%S")
    render_path = os.path.join(save_dir,
                               f"{render_fname}_{fidxs[0]}-{fidxs[-1]}_{timestr}.mp4")

    if save_frames:
        TMP_DIR = os.path.join(save_dir, 'tmp', 'render')
        # Clear temp dir for the frames
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        os.makedirs(TMP_DIR, exist_ok=True)
        print(f"Saving images to: {TMP_DIR}")

    # Load sleap data and read in skeleton info
    clip = sleap.load_file(f'{expt_path}/inference.cleaned.proofread.slp')
    skeleton = clip.skeleton

    if type(lims) == str and lims == "fit":  # TODO: define what lims do clearly
        pad = np.array([-1, 1]) * padding
        xl, yl = get_lims(trx)
        xl = xl + pad
        yl = yl[::-1] + pad
        print(xl, yl)
    elif lims is None:
        xl, yl = None, None
    else:
        xl, yl = lims

    if render_stride > 1:
        trx = trx[::render_stride]
        fidxs = fidxs[::render_stride]

    render_t0 = time()
    if save_frames:
        _iterator = np.arange(len(fidxs))
        _iterator = rich.progress.track(_iterator)
    else:
        _iterator = [0]

    img = 255 + np.zeros_like(clip.video[0].squeeze())

    for t in _iterator:
        # Get frame data
        fidx = fidxs[t]
        poses = trx[fidx].reshape(-1, 2)

        # Plot image and zoom
        fig = imgfig(img, figsize=(fig_size_px / dpi, fig_size_px / dpi), dpi=dpi)
        if xl is not None:
            plt.xlim(xl)
        if yl is not None:
            plt.ylim(yl)

        # Plot poses
        # plot_skel(poses, skeleton, full_skel=False, c=cmap[0], zorder=20, ms=ms, lw=lw, alpha=alpha)
        plot_skel(poses, skeleton, full_skel=draw_skel, nodes=draw_nodes, c='k', zorder=20, ms=ms, lw=lw, alpha=alpha)

        # Overlay the positions of render_node_names nodes
        if render_nodes == "all":
            node_cmaps = cmap
            render_trx = poses
            plt.scatter(render_trx[:, 0], render_trx[:, 1], c=node_cmaps, s=12)
        else:
            node_cmaps = cmap[render_nodes]
            render_trx = poses[render_nodes, :]
            # node_alphas = np.zeros()
            plt.scatter(render_trx[:, 0], render_trx[:, 1], c=node_cmaps, s=12)

        if save_frames:
            # Render frame
            plt.savefig(f"{TMP_DIR}/frame.{t:04}.png", dpi=dpi)
            plt.close(fig)

    if save_frames:
        # Render frames into video
        # !ffmpeg -y -framerate {fps} -i "{TMP_DIR}/frame.%04d.png" -c:v libx264 -preset superfast -pix_fmt yuv420p -crf {crf} "{TMP_DIR}/movie.mp4" >nul 2>nul
        os.system(f"ffmpeg -y -framerate {fps} -i {os.path.join(TMP_DIR, 'frame.%04d.png')} -c:v libx264 -preset superfast -pix_fmt yuv420p -crf {crf} {os.path.join(TMP_DIR, 'movie.mp4')}")
        shutil.copy(f"{os.path.join(TMP_DIR, 'movie.mp4')}", render_path)
        print(f"Rendered movie: {TMP_DIR}")
        print(f"Copied to: {render_path}")

        # Clear tmp directory
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    render_dt = time() - render_t0
    print(f"Rendered in {render_dt:.1f} s")

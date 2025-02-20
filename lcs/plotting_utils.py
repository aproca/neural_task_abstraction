from contextlib import contextmanager
import jax
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.transforms import blended_transform_factory
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter1d, gaussian_filter1d
from sklearn.decomposition import TruncatedSVD
from IPython.display import HTML
import matplotlib.animation as animation
from lcs.init_mpl import init_mpl
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from scipy import stats
from matplotlib import patches as mpatches
from matplotlib import lines as mlines
import logging
from lcs.utils import compute_similarity
logger = logging.getLogger(__name__)

limpad = .1

def fractions(x,pos, step):
    if np.isclose((x/step)%(1./step),0.):
        # x is an integer, so just return that
        return '{:.0f}'.format(x)
    else:
        # this returns a latex formatted fraction
        return rf"${'+' if x > 0 else '-'}$"+'$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(np.abs(x/step),1./step)
        # if you don't want to use latex, you could use this commented
        # line, which formats the fraction as "1/13"
        ### return '{:2.0f}/{:2.0f}'.format(x/step,1./step)


def format_axis(ax, line_width_multiplier=2, font_size_multiplier=1, plot_line_multiplier=1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.xaxis.set_tick_params(size=8*font_size_multiplier)
    ax.yaxis.set_tick_params(size=8*font_size_multiplier)

    ## SET AXIS WIDTHS
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2*line_width_multiplier)

    # increase tick width
    ax.tick_params(width=2*line_width_multiplier)

    ax.xaxis.label.set_fontsize(16*font_size_multiplier)
    ax.yaxis.label.set_fontsize(16*font_size_multiplier)

    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(14*font_size_multiplier)

    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(14*font_size_multiplier)

    for line in ax.lines:
        line.set_linewidth(2.5*plot_line_multiplier)

    if ax.get_legend() is not None:

        for legobj in ax.get_legend().legendHandles:
            legobj.set_linewidth(2.0*line_width_multiplier)

    ax.title.set_fontsize(16*font_size_multiplier)
    # ax.title.set_fontsize(24*font_size_multiplier)

def fill_between(
        ax,
        x,
        y=None,
        y_mean=None,
        y_std=None,
        gauss_reduce=True,
        line=True,
        discrete=False,
        use_sem = True,
        **line_args,
):
    if gauss_reduce:
        if y is not None:
            fac = y.shape[0] ** 0.5
        else:
            fac = gauss_reduce
    else:
        fac = 1

    fill_alpha = line_args.pop("fill_alpha", .3)
    line_alpha = line_args.pop("alpha", 1.)

    if (y is not None) and (y.shape[0] == 1):
        l, = ax.plot(x, y[0], alpha=line_alpha, **line_args)
        return l, None

    if y is not None:
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            # leave immediately
            l, = ax.plot(x, y[0], **line_args)
            return l, None
        mean = y.mean(axis=0)
        sem = stats.sem(y, axis=0) / fac
        std = y.std(axis=0) / fac
        if (std < 1e-10).all(): logger.warning("Trivial std observed while attempting fill_between plot")
    else:
        mean = y_mean
        std = y_std / fac

    
    if not discrete:
        if line:
            line_args["markersize"] = 4 if not line_args.get("markersize") else line_args["markersize"]
            (l,) = ax.plot(x, mean, alpha=line_alpha, **line_args)
            lc = l.get_color()
        else:
            l = None
            lc = None

        if use_sem:
            if (sem != 0).any():
                c = line_args.get("color", lc)
                idx = np.argsort(x)
                x = np.array(x)
                fill = ax.fill_between(
                    x[idx],
                    (mean - sem)[idx],
                    (mean + sem)[idx],
                    alpha=fill_alpha,
                    color=c,
                    zorder=-10,
                )
        else:
            if (std != 0).any():
                c = line_args.get("color", lc)
                idx = np.argsort(x)
                x = np.array(x)
                fill = ax.fill_between(
                    x[idx],
                    (mean - std)[idx],
                    (mean + std)[idx],
                    alpha=fill_alpha,
                    color=c,
                    zorder=-10,
                )
            else:
                fill = None
    else:
        ls = line_args.pop("ls", "none")
        marker = line_args.pop("marker", "o")

        if use_sem:
            l = ax.errorbar(
                x, mean, yerr=sem, ls=ls, fmt=marker, capsize=4, **line_args
            )
        else:
            l = ax.errorbar(
                x, mean, yerr=std, ls=ls, fmt=marker, capsize=4, **line_args
            )
        fill = None

    return l, fill

def init_plot_settings():
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams['axes.spines.top'] = False

def plot_loss(ax, tape, cfg, skip=1, gauss_reduce=True, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    fill_between(ax, tape.t.mean(0), tape.loss, label='loss', **(dict(color="k", alpha=1.) | largs), gauss_reduce=gauss_reduce)
    # fill_between(ax, t, tape.loss_mono, label='loss mono', color="tab:red", alpha=.5, zorder=-10)

    ax.set_ylabel(r"loss $\mathcal{L}$")

    ax.set_yticks([0, .5, 1])
    ax.dataLim.y0 = 0.
    ax.dataLim.y1 = 1.1
    ax.autoscale_view()

def plot_norm_w(ax, tape, cfg, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    t = tape.t.mean(0)

    if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono':
        fill_between(ax, t, tape.norm_W1.mean((-1)), **(dict(label='norm of W', c="k") | largs)) 
        if cfg.control == 'deep_mono':
            fill_between(ax, t, tape.norm_W2.mean((-1)), **(dict(label='norm of W', c="tab:red") | largs)) 
    else:
        fill_between(ax, t, tape.norm_W1.mean((-1, -2)), **(dict(label='norm of W', c="k") | largs)) 
    ax.set_ylabel(r'$|W^p|$', color='k')

    # ax.dataLim.y0 = 0
    # ax.autoscale_view()

def plot_cos_similarity(ax, tape, t, cfg, pad, compute_sim=None, **largs):
    if tape.t.ndim == 1:
        tape = jax.tree.map(lambda x: x[None], tape)
    
    # First, let's abstract the similarity computation
    def get_similarity(tape, cfg):
        if compute_sim is not None:
            sim = compute_sim(tape.W_teachers[:, None, :, None],  # (B, 1, M, 1, ...)
                               tape.W1[:, :, None, :]             # (B, T, 1, P, ...)
                               )
            return sim  # (B, T, M, P, ...)
        
        if cfg.control in ['N_diag_mono', '2_diag_mono', 'deep_mono']:
            return tape.sorted_SVD_sim if cfg.teacher_mode == 'svd' else tape.sorted_cossim
        else:
            return tape.SVD_similarity if cfg.teacher_mode == 'svd' else tape.cos_sim1
    
    # Get the appropriate similarity measure
    sim = get_similarity(tape, cfg)
    
    # Now plot using the computed similarity
    for p in range(cfg.num_paths):
        for m in range(cfg.num_contexts):
            t_pad = (-2**p)*pad if p == m else 0
            l, fill = fill_between(ax, t, sim[..., m, p] + t_pad, color=f'C{p}', alpha=1, ls='dashed')
            l.set_gapcolor(f'C{m}')
    
    ax.set_ylabel('SVD similarity' if cfg.teacher_mode == 'svd' else 'Cosine similarity')



def plot_dynamics(ax, tape, cfg, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    skip = int(max(cfg.T_tot // cfg.T_tape, 1))
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))
    t = np.atleast_1d(tape.t).mean(0)

    if cfg.control == 'deep_mono':
        delta_W2 = np.linalg.norm(np.gradient(tape.W2, axis=1), axis=-1).mean((-1))
        delta_W2 = maximum_filter1d(delta_W2, size=window, mode='nearest', axis=1)
        fill_between(ax, t, delta_W2, color='red', ls="-", **largs)
    else:
        delta_c = np.gradient(tape.c1[...,-1], axis=1)
        delta_c = maximum_filter1d(delta_c, size=window, mode='nearest', axis=1)
        fill_between(ax, t, delta_c, color='gray', ls="-", **largs)
    
    if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono':
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-1))
    else:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-2, -1))
    delta_W = maximum_filter1d(delta_W, size=window, mode='nearest', axis=1)
    fill_between(ax, t, delta_W, color='k', ls="-", **largs)

def plot_context_alignment(ax, tape, t, cfg, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    if cfg.teacher_mode == 'svd':
        fill_between(ax, t, tape.SVD_alignment, color='k', alpha=1, **largs)
        ax.set_ylabel("SVD alignment")
    else:
        fill_between(ax, t, tape.context_alignment1, color='k', alpha=1, **largs)
        ax.set_ylabel("Alignment")

def make_teacher_arrows(ax, s=3, max_=1., **largs):
    
    # make arrows
    kwargs = dict(head_width=0.15*s, head_length=0.15*s, width=.05*2, zorder=-10, alpha=.2, length_includes_head=True) | largs
    ax.arrow(0, 0, max_, 0,  fc="C0", ec="none",  **kwargs)
    ax.arrow(0, 0, 0, max_, fc="C1", ec="none", **kwargs)

def plot_c(ax, tape, cfg, swap=False, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    alpha = .75
    t = tape.t.mean(0)
    if cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono':
        ax.set_ylabel("Sorted $c^p$")
    else:
        ax.set_ylabel("gates $c^p$")

    if cfg.control == 'N_diag_mono':
        for p in range(cfg.num_paths):
            fill_between(ax, t, np.mean(tape.sorted_c_student[:,:,p],axis=(2)), **(dict(color=f'C{p}', alpha=alpha) | largs), use_sem=True)
    elif cfg.control == 'deep_mono':
        for p in range(cfg.num_paths):
            fill_between(ax, t, np.mean(tape.sorted_W2_student[:,:,p],axis=(2,3)), **(dict(color=f'C{p}', alpha=alpha) | largs), use_sem=True)
    else:
        if cfg.num_paths > 3:
            for p in range(cfg.num_paths):
                ax.plot(t, np.mean(tape.c1[...,p],axis=0), color=f'C{p}', alpha=alpha)
        else:
            for p in range(cfg.num_paths):
                if tape.c1.ndim == 3:
                    fill_between(ax, t, tape.c1[..., p] if not swap else tape.c1[..., ::-1][..., p], **(dict(color=f'C{p}', alpha=alpha) | largs),  use_sem=True)
                else:
                    # c in each path is vector-valued
                    for i in range(tape.c1.shape[-1]):
                        fill_between(ax, t, tape.c1[..., p, i] if not swap else tape.c1[..., ::-1][..., p, i], **(dict(color=f'C{p}', alpha=.4) | largs),  use_sem=True)

                    

    if cfg.control == 'N_diag_mono' or cfg.control == '2_diag_mono' or cfg.context_model:
        c_min = np.min(tape.c1)
        c_max = np.max(tape.c1)
    elif cfg.control == 'deep_mono':
        c_min = np.min(tape.sorted_W2_student)
        c_max = np.max(tape.sorted_W2_student)
    else:
        c_min = np.min(tape.c1)
        c_max = np.max(tape.c1)

        
    ax.dataLim.y0 = c_min - limpad 
    ax.dataLim.y1 = c_max + limpad 
    ax.set_yticks([0, .5, 1, 1.5, 2, 2.5])
    ax.autoscale_view()

def xylabel_to_ticks(ax, which="both", pad=0.):
    fig = ax.get_figure()
    fig.canvas.draw()

    if which == "both":
        which = "all"

    if which == "x":
        which = "bottom"

    if which == "y":
        which = "left"

    if which == "all":
        for which_ in ["left", "bottom"]:
            xylabel_to_ticks(ax, which=which_, pad=pad)

    if which == "top" or which == "bottom":
        x_label = ax.xaxis.get_label()
        
        ax.xaxis.get_label().set_horizontalalignment("center")
        ax.xaxis.get_label().set_verticalalignment("bottom" if which == "top" else "top")
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        x_label_coords = trans.inverted().transform(ax.transAxes.transform(x_label.get_position()))

        ax.xaxis.set_label_coords(x_label_coords[0], (0 if which == "bottom" else 1) + pad, transform=trans)

    if which == "left" or which == "right":
        y_label = ax.yaxis.get_label()
        
        ax.yaxis.get_label().set_horizontalalignment("center")
        ax.yaxis.get_label().set_verticalalignment("bottom" if which == "left" else "top")
        ticklab = ax.yaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()

        y_label_coords = trans.inverted().transform(ax.transAxes.transform(y_label.get_position()))
        ax.yaxis.set_label_coords((0 if which == "left" else 1) + pad, y_label_coords[1], transform=trans)

# Define the colors
colors = ["tab:red", "white", "tab:green"]

# Create the colormap
cmap_div = LinearSegmentedColormap.from_list("custom_diverging", colors, N=512)

def get_w_h(path):
    if path.suffix == ".svg":
        import xml.etree.ElementTree as ET

        svg = ET.parse(path).getroot().attrib
        import re

        w = svg["width"]
        h = svg["height"]
        w = float(re.sub("[^0-9]", "", w))
        h = float(re.sub("[^0-9]", "", h))
    elif path.suffix == ".pdf":
        from PyPDF2 import PdfFileReader

        input1 = PdfFileReader(open(path, "rb"))
        mediaBox = input1.getPage(0).mediaBox
        w, h = mediaBox.getWidth(), mediaBox.getHeight()
    else:
        raise NotImplementedError
    return w, h

def place_graphic(ax, inset_path, fit=None, mode="raster", inkscape_kwargs={}):
    import subprocess
    from pathlib import Path
    import platform
    fig = ax.get_figure()
    plt.rcParams['text.usetex'] = False
    ax.cla()
    ax.axis("off")
    # no_spine(ax, which="right", remove_all=True)

    # freeze fig to finish off layout, new in 3.6
    fig.canvas.draw()
    fig.set_layout_engine('none')

    ax_bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()

    plt.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "pgf.preamble": r"\usepackage{graphicx}\usepackage[export]{adjustbox}\usepackage{amsmath}",
        }
    )

    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    import matplotlib

    # TeX rendering does only work if saved as pdf
    matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)

    bbox = {"width": ax_bbox.width * fig_w, "height": ax_bbox.height * fig_h}

    import tempfile, shutil, os

    def create_temporary_copy(path):

        temp_dir = Path(
            "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()
        )
        rand_seq = np.random.choice(["a", "b", "c", "d", "e"], size=10)
        temp_path = os.path.join(temp_dir, f'{"".join(rand_seq)}{path.suffix}')
        shutil.copy2(path, temp_path)
        return temp_path, temp_dir

    path_str, temp_dir = create_temporary_copy(inset_path)
    if inset_path.suffix == ".svg":
        # we first need to convert to a format that allows us to embed
        inkscape_kwargs_dflt = {}
        if mode == "raster":
            inkscape_kwargs_dflt = inkscape_kwargs_dflt | {'export-dpi': 300}
            path_str_rendered = str(temp_dir / (inset_path.stem + ".png"))
            command = [
                "inkscape",
                path_str,
                f"--export-filename={path_str_rendered}",
            ]
        else:
            path_str_rendered = str(temp_dir / (inset_path.stem + ".pdf"))
            command = [
                "inkscape",
                path_str,
                f"--export-filename={path_str_rendered}"
            ]
        command.extend([f"--{key}" if value is None else f"--{key}={value}" for key, value in inkscape_kwargs.items()])

        p = subprocess.run(command, capture_output=True, text=True)
        path_str = path_str_rendered

        # print inkscape --help if failed
        if p.returncode != 0:
            print(p.stderr)
            help_out = subprocess.run(["inkscape", "--help"], capture_output=True, text=True)
            print(help_out.stdout)
            
    if fit is None:
        w, h = get_w_h(inset_path)
        if w / h > bbox["width"] / bbox["height"]:
            fit = "width"
        else:
            fit = "height"
    else:
        assert "width" in fit or "height" in fit
        
    if path_str.endswith(".pdf"):
        # embed via LaTeX – quite buggy!
        tex_cmd = ""
        tex_cmd += r"\centering"
        tex_cmd += rf"\includegraphics[{fit}={{{bbox[fit]:.5f}in}}]{{{path_str}}}"
        print(bbox[fit])
        ax.text(0.0, 0.0, tex_cmd)
    else:
        # embed via imshow
        ax.imshow(plt.imread(path_str))




def plot_2D_W(ax, W_s__proj, alpha_min=.3, alpha_max=1., make_arrows=False, sv_max=10, tr=None, off=True, **largs):
    """
    W_S_proj: (t p c a) = (time, paths, contexts, svs)
    """
    
    if off:
        ax.axis('off') 
    ax.set_box_aspect(1)

    tr = tr if tr is not None else lambda x: x

    if sv_max > W_s__proj.shape[-1]:
        sv_max = W_s__proj.shape[-1]

    num_students = W_s__proj.shape[2]
    for sv in range(sv_max):
        for p in range(num_students):
            x, y = W_s__proj[:, 0, p, sv], W_s__proj[:, 1, p, sv]
            plot_traj(ax, tr(x), tr(y), alpha_min=alpha_min, alpha_max=alpha_max, **(dict(c=f"C{p}" if num_students > 1 else 'k') | largs))

    # mark the teachers
    if make_arrows:
        make_teacher_arrows(ax)

    return

def plot_2D_c(ax, tape, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    c = tape.c1[0] # (T, P), get first in batch
    
    ax.set_box_aspect(1)
    

    # no top or right axis spines visibility, but keep left and bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plot_traj(ax, c[..., 0], c[..., 1], **(dict(c='k', alpha_min=.3, alpha_max=1.,) | largs))

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    
    ax.dataLim.y0 = 0 
    ax.dataLim.y1 = 1
    ax.dataLim.x0 = 0
    ax.dataLim.x1 = 1

    ax.margins(0.05)

    ax.autoscale_view()

    return

def indicate_contexts(ax, tape, cfg, swap=False, use_tape=False, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    import seaborn as sns
    colors = sns.color_palette("deep", as_cmap=True)
    cmap = LinearSegmentedColormap.from_list('ctx_colors', colors if not swap else colors[::-1])

    t = tape.t.mean(0)

    if cfg.Y_tgt is None and not hasattr(cfg.c_gt_curriculum, '__call__') and not use_tape:
        color_indcs = np.zeros_like(t).astype(int)
        
        for i, t_switch in enumerate(np.arange(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
            switch_point = int(0.75*cfg.num_blocks)
            if cfg.shared_teachers or cfg.shared_concat_teachers:
                if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B__AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = 'C2'
                elif cfg.c_gt_curriculum == 'AB_BC__':
                    c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_AB__':
                    c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = f'C{(i % cfg.num_contexts)+3}'
                elif cfg.c_gt_curriculum == 'AB_BC_CA__A_B_C__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)+cfg.num_contexts}'
                    else:
                        c = f'C{(i % cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                    c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded
                    if i < switch_point:
                        if i % 2 == 0:
                            c = 'C4'
                        else:
                            c = 'C6'
                    else:
                        c = 'C7'
                elif cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': # hard-coded solution
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + cfg.num_contexts}'
                    else:
                        c = f'C{(i % 2) + 2*cfg.num_contexts}'
            if cfg.c_gt_curriculum == 'A_B__':
                c = f'C{i % cfg.num_contexts + (0 if not swap else 1)}'

            color_indcs[(t >= t_switch) & (t < t_switch+cfg.block_duration)] = int(c[1:])

            # ax.axvspan(t_switch, t_switch+cfg.block_duration, color=c, alpha=0.1, zorder=-10)
    else: 
        # get a *float* scalar color score for every time, sth like Y_tgt in [0, 7], allowing for smooth transitions
        
        if cfg.Y_tgt is not None:
            assert cfg.Y_tgt(t).shape[-1] == cfg.num_contexts
            assert np.allclose(cfg.Y_tgt(t).sum(-1).mean(), 1, atol=5e-2)
            color_indcs = (cfg.Y_tgt(t)*np.arange(cfg.num_contexts)).sum(-1)  
        else:
            # vector of form (.9, .1) that sums to one
            if not hasattr(tape, 'c_gt1'):
                c_gt_vec = cfg.c_gt_curriculum(t)
            else:
                c_gt = (tape.c_gt1[0] if tape.c_gt1.ndim == 2 else tape.c_gt1).astype(int)
                c_gt_vec = np.eye(cfg.num_contexts)[c_gt]
            c_vals = np.arange(cfg.num_contexts + cfg.num_shared_contexts)
            p_c_gt_vec = c_gt_vec / c_gt_vec.sum(-1, keepdims=True)  # density
            color_indcs = (p_c_gt_vec * c_vals).sum(-1)  # will give float in [0, num_contexts-1]

    # normalize to [0, 1] for colormap
    color_indcs = color_indcs / len(colors)

    with no_autoscale(ax):
        transform = blended_transform_factory(ax.transData, ax.transAxes)
        ax.pcolormesh(t, [0, 1], np.repeat(color_indcs[:, None], 2, axis=1).T, cmap=cmap, vmin=0, vmax=1, zorder=-10, rasterized=True, transform=transform, **(dict(alpha=0.25) | largs))

def setup_axes_labels(axd, labels=None):
    labels = [rf"$\mathbf{{{k.upper() if labels is None else labels[i_k]}}}$" for i_k, k in enumerate(list(axd.keys()))]
    for i, ax in enumerate(axd.values()):
        ax.text(x=-.2, y=1.2, s=labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')

def make_ticks(ax, cfg, skip=1, minor=False):
    # import major and minor locator
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter, FixedFormatter, FuncFormatter
    majorLocator = MultipleLocator(skip*cfg.W_tau)
    minorLocator = MultipleLocator(cfg.c_tau)

    # set major and minor locator
    ax.xaxis.set_major_locator(majorLocator)
    if minor:
        ax.xaxis.set_minor_locator(minorLocator)

    # just count every W_tau
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: (np.round(x / cfg.W_tau).astype(int) if np.round(x / cfg.W_tau).astype(int) % skip == 0 else '')))

def plot_tape_activity(fig, tape, cfg):
    init_plot_settings()

    skip = int(max(cfg.T_tot // cfg.T_tape, 1))
    # skip = max(cfg.num_blocks * cfg.block_duration // tape.t.shape[-1], 1)
    if fig is None:
        TEXTWIDTH = 7.3  # inches
        TEXTHEIGHT = 10.5  # inches
        fig = plt.figure(figsize=(TEXTWIDTH / 3 * 2, TEXTHEIGHT / 2.), layout="constrained")
    axd = fig.subplot_mosaic('''
    0
    a
    b
    c
    d
    e
    f
    ''')
    plot_loss(axd["a"], tape, cfg, skip)
    plot_norm_w(axd["b"], tape, tape.t.mean(0), cfg)
    plot_cos_similarity(axd["d"], tape, tape.t.mean(0), cfg, 0.02)
    plot_dynamics(axd["e"], tape, tape.t.mean(0), cfg, skip)
    plot_context_alignment(axd["f"], tape, tape.t.mean(0), cfg)
    setup_axes_labels(axd)
    fig.align_labels()
    return fig


def make_cosyne_fig(tape, cfg, fig=None, W_teachers=None, fast=True, alignment='default', compute_sim=None):
    # TODO: implement with plotting from monolithic model; make compatable for several layers
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    skip = int(max(cfg.T_tot // cfg.T_tape, 1))
    # skip = max(cfg.num_blocks * cfg.block_duration // tape.t.shape[-1], 1)
    # learning_rate_c_eff = tape.learning_rate_c_eff.mean(-1)
    # svs = np.linalg.svd(tape.W1)

    if fig is None:
        TEXTWIDTH = 7.3  # inches
        TEXTHEIGHT = 10.5  # inches
        fig = plt.figure(figsize=(TEXTWIDTH/3*2, TEXTHEIGHT / 2.), layout="constrained" if not fast else "none")
    
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    axd = fig.subplot_mosaic('''
    a
    b
    c
    d
    e
    f
    g
    ''')

    fig.axd = axd

    # unshare x for 0 and a
    for k, v in axd.items():
        if k != "0":
            v.sharex(axd["a"])
            # v.sharey(axd["0"])

    if W_teachers is not None:
        try:
            plot_embed(axd["0"], tape, cfg, W_teachers)
        except ValueError:
            pass


    from lcs.plotting_utils import fill_between
    from functools import partial
    fill_between = partial(fill_between, gauss_reduce=True, line=True)
    t = tape.t.mean(0)
    print(t.min(), t.max())

    # %%
    ax = axd["a"]

    plot_loss(ax, tape, cfg, skip)


    # %%
    # norm of W
    ax = axd["b"]
    plot_norm_w(ax, tape, cfg)

    # ax_tw.plot(tape.W[:, 0, 1], label='W i = 1', c="C2", ls=":")
    # ax_tw.plot(tape.W[:, 1, 1], label='W i = 2', c="C2", ls="--")


    # %%
    # norm of c
    ax = axd["c"]
    plot_c(ax, tape, cfg)
    ax.dataLim.y0 = 0
    ax.dataLim.y1 = 1
    ax.autoscale_view()


    # teacher 0, student 1
    pad = .02

    ax = axd["d"]

    plot_cos_similarity(ax, tape, tape.t.mean(0), cfg, 0.02, compute_sim=compute_sim)
    # plot_similarity(ax, tape, cfg)
    ax.dataLim.y0 = 0
    ax.dataLim.y1 = 1
    ax.autoscale_view()

    ax = axd["e"]
    plot_delta(ax, tape, cfg,)

    ax = axd["f"]
    if alignment == 'default':
        l, fill = fill_between(ax, t, tape.context_alignment1, color='k', label="context alignment", alpha=1)
        l.set_gapcolor("C0")

    else:
        # this alignment metric measures whether *some* singular vector is aligned
        W_teachers, W_students = tape.W_teachers[0], tape.W1[0]
        K_U = compute_similarity(W_teachers[:, None], W_students[:, None, :], metric='svd,u')  # (T, C, P)
        ax.plot(t, K_U[:, 0, 0], c='C0', label='$u^0 \cdot \hat u^0_:$', alpha=1.)
        ax.plot(t, K_U[:, 1, 0], c='C0', label='$u^1 \cdot \hat u^0_:$', alpha=.5)
        ax.plot(t, K_U[:, 0, 1], c='C1', label='$u^0 \cdot \hat u^1_:$', alpha=.5)
        ax.plot(t, K_U[:, 1, 1], c='C1', label='$u^1 \cdot \hat u^1_:$', alpha=1.)
        ax.legend()

        # this alignment metric measures whether *some* singular vector is aligned
        W_teachers, W_students = tape.W_teachers[0], tape.W1[0]
        ax = axd["g"]
        K_V = compute_similarity(W_teachers[:, None], W_students[:, None, :], metric='svd,v')  # (T, C, P)
        ax.plot(t, K_V[:, 0, 0], c='C0', label='$v^0 \cdot \hat v^0_:$', alpha=1.)
        ax.plot(t, K_V[:, 1, 0], c='C0', label='$v^1 \cdot \hat v^0_:$', alpha=.5)
        ax.plot(t, K_V[:, 0, 1], c='C1', label='$v^0 \cdot \hat v^1_:$', alpha=.5)
        ax.plot(t, K_V[:, 1, 1], c='C1', label='$v^1 \cdot \hat v^1_:$', alpha=1.)
        ax.dataLim.y0 = 0
        ax.dataLim.y1 = 1
        ax.autoscale_view()
        ax.legend()

    axd["f"].dataLim.y0 = 0 - limpad
    axd["f"].dataLim.y1 = 1 + limpad
    axd["f"].autoscale_view()

    axd["f"].set_ylabel("Alignment")

    for k, ax in axd.items():
        indicate_contexts(ax, tape, cfg)


    fig.align_labels()
    fig.axd = axd

    return fig

def plot_delta(ax, tape, cfg, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    t = tape.t.mean(0)

    delta_c =  np.gradient(tape.c1[...,-1], axis=1) #TODO: make flexible for layer sizes
    # diff along time, then vector norm, then mean over paths and features
    if (cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono') and cfg.context_model == False:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-1)) #TODO: make flexible for layer sizes
    else:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-2, -1)) #TODO: make flexible for layer sizes
    # delta_W_mono = np.linalg.norm(np.gradient(tape.W_mono, axis=1), axis=-1).mean(-1)

    skip = 1

    # make moving average with window n_repeats
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    ax_tw = ax.twinx()
    if delta_c.ndim == 3:
        fill_between(ax_tw, t, delta_c.mean(-1), **(dict(color='gray', ls="--") | largs))
    else:
        fill_between(ax_tw, t, delta_c, **(dict(color='gray', ls="--") | largs))
    # axd["e"].set_yscale("log")

    # ax_tw.set_yscale("log")
    fill_between(ax, t, delta_W, **(dict(color='k', ls="-") | largs))
    # fill_between(ax_tw, t, delta_W_mono, color='tab:red', ls="-", zorder=-10)

    ax_tw.dataLim.y0 = 0
    ax.dataLim.y0 = 0.
    ax.autoscale_view()

    # ylable
    ax.set_ylabel(r"weight change" + '\n' + r"$\langle |\Delta W| \rangle$", color='k')
    ax_tw.set_ylabel(r"gate change" + '\n' + r"$\langle \Delta c \rangle$", color='gray', rotation='horizontal', ha='left', va="center")

    ax.set_xlabel("time $t$")
    ax_tw.set_zorder(ax.get_zorder()+1)
    # right spine
    ax_tw.spines['right'].set_visible(True)

def plot_delta_W(ax, tape, cfg, which="W1", normalized=True, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    t = tape.t.mean(0)
    # diff along time, then vector norm, then mean over paths and features

    which1 = which[:2]
    which2 = which[2:]
    
    W = getattr(tape, which1)
    W = W / np.linalg.norm(W, axis=(-1, -2), keepdims=True)  # normalize by norm of W
    delta_W = np.linalg.norm(np.gradient(W, axis=1), axis=(-1, -2))  # mean over contexts
    delta_W = delta_W.mean(-1) if delta_W.ndim == 3 else delta_W  # btp -> bt if path dim is still present

    if len(which2) > 0:
        W = getattr(tape, which2)
        W = W / np.linalg.norm(W, axis=(-1, -2), keepdims=True)  # normalize by norm of W
        delta_W2 = np.linalg.norm(np.gradient(W, axis=1), axis=(-1, -2))  # mean over contexts
        delta_W2 = delta_W.mean(-1) if delta_W.ndim == 3 else delta_W  # btp -> bt if path dim is still present

        delta_W = np.mean([delta_W, delta_W2], axis=0)

    
    if normalized:
        lb = delta_W.shape[1] // 2
        delta_W = delta_W / np.max(delta_W[:, -lb:], axis=-1, keepdims=True)

    skip = 1

    # make moving average with window n_repeats
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    # ax_tw.set_yscale("log")
    fill_between(ax, t, delta_W, **(dict(color='k', ls="-") | largs))
    # fill_between(ax_tw, t, delta_W_mono, color='tab:red', ls="-", zorder=-10)

    ax.dataLim.y0 = 0.
    ax.autoscale_view()

    # ylable
    ax.set_ylabel(r"weight change" + '\n' + r"$||d/dt \: \boldsymbol{W} ||$" + ('' if not normalized else ' (rel.)'), color='k')



def plot_delta_c(ax, tape, cfg, **largs):
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape

    t = tape.t.mean(0)

    delta_c =  np.linalg.norm(np.gradient(tape.c1[...], axis=1), axis=-1)

    skip = 1

    # make moving average with window n_repeats
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    if delta_c.ndim == 2:
        delta_c = delta_c[..., None]
    for i in range(delta_c.shape[-1]):
        fill_between(ax, t, delta_c[..., i], **(dict(color=f'k', ls="-") | largs))
    # axd["e"].set_yscale("log")

    ax.dataLim.y0 = 0
    ax.dataLim.y0 = 0.
    ax.autoscale_view()

    # ylable
    ax.set_ylabel(r"gate change" + '\n' + r"$||d/dt \: \boldsymbol{c} ||$", color='k')
    ax.set_zorder(ax.get_zorder()+1)

def plot_similarity(ax, tape, cfg, sim=None, **largs):
    
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    pad = .02
    t = np.atleast_1d(tape.t).mean(0)
    dflt = dict(alpha=1, ls="dashed", label="cosine similarity")

    if cfg.control == 'N_diag_mono' or cfg.control == '2_diag_mono' or cfg.control == 'deep_mono':
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c and cfg.num_paths <= 3:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                if cfg.teacher_mode == 'svd':
                    l, fill = fill_between(ax, t, tape.sorted_SVD_sim[...,c,p] + t_pad, color = f'C{p}', alpha=1, ls='dashed', use_sem=True)
                else:
                    l, fill = fill_between(ax, t, tape.sorted_cossim[...,c,p] + t_pad, color = f'C{p}', alpha=1, ls='dashed',  use_sem=True)
                l.set_gapcolor(f'C{c}')

    else:
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                if cfg.teacher_mode == 'svd':
                    l, fill = fill_between(ax, t, tape.SVD_similarity[..., c, p]+t_pad, color=f'C{p}', alpha=1, ls="dashed", use_sem=True)
                else:
                    l, fill = fill_between(ax, t, tape.cos_sim1[..., c, p]+t_pad, color=f'C{p}', alpha=1, ls="dashed", use_sem=True)
                l.set_gapcolor(f'C{c}')

    ax.dataLim.y0 = 0 - limpad
    ax.dataLim.y1 = 1 + limpad

    ax.set_ylabel(r"$\frac{W^p \cdot W^{p'}}{|W^p| \cdot |W^{p'}|}$")

@contextmanager
def no_autoscale(ax=None, axis="both"):
    ax = ax or plt.gca()
    ax.figure.canvas.draw()
    lims = [ax.get_xlim(), ax.get_ylim()]
    yield
    if axis == "both" or axis == "x":
        ax.set_xlim(*lims[0])
    if axis == "both" or axis == "y":
        ax.set_ylim(*lims[1])

def c_line(ax, x, y, cmap=None, c=None, colors=None, **largs):
    if cmap:
        colors = cmap(np.arange(len(x)))
    if colors is not None and c is None:
        colors = colors
    if c is not None and colors is None:
        colors = c

    c = colors

    c = mcolors.to_rgba(c) if type(c) == str else c
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ls = largs.pop('ls', '-')
    lw = largs.pop('lw', 1)
    alpha = largs.pop('alpha', None)
    zorder = largs.pop('zorder', -5)
    cmap = ListedColormap([(mcolors.to_rgba(color)[:3], alpha) if alpha is not None else mcolors.to_rgba(color) for color in c])
    if ls != 'none':
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        color_val = np.linspace(0, 1, len(x))
        # Set the values used for colormapping
        lc.set_array(color_val)
        lc.set_linewidth(lw)
        lc.set_zorder(zorder)
        lc.set_linestyle(ls)
        line = ax.add_collection(lc, autolim=True)
        ax.autoscale_view()

    return line

def add_arrow_to_line2D(
    ax, line, arrow_locs=[0.2, 0.4, 0.6, 0.8],
    arrowstyle='-|>', arrowsize=1, transform=None):
    """
    Add arrows to a matplotlib.lines.Line2D at selected locations.

    Parameters:
    -----------
    axes: 
    line: list of 1 Line2D obbject as returned by plot command
    arrow_locs: list of locations where to insert arrows, % of total length
    arrowstyle: style of the arrow
    arrowsize: size of the arrow
    transform: a matplotlib transform instance, default to data coordinates

    Returns:
    --------
    arrows: list of arrows
    """
    if (not(isinstance(line, list)) or not(isinstance(line[0], 
                                           mlines.Line2D))):
        raise ValueError("expected a matplotlib.lines.Line2D object")
    x, y = line[0].get_xdata(), line[0].get_ydata()

    arrow_kw = dict(arrowstyle=arrowstyle, mutation_scale=10 * arrowsize)
    if transform is None:
        transform = ax.transData

    arrows = []
    for loc in arrow_locs:
        s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        n = np.searchsorted(s, s[-1] * loc)
        arrow_tail = (x[n], y[n])
        arrow_head = (np.mean(x[n:n + 2]), np.mean(y[n:n + 2]))
        p = mpatches.FancyArrowPatch(
            arrow_tail, arrow_head, transform=transform, zorder=10,
            **arrow_kw)
        ax.add_patch(p)
        arrows.append(p)
    return arrows


def plot_traj(ax, x, y, alpha_min=.3, alpha_max=1., mark='none', gain=None, **largs):
    if gain is None:
       gain = lambda t: t

    c = largs.pop('c', largs.pop('color', 'k'))
    alpha = largs.pop('alpha', 1)
    lw = largs.pop('lw', 1)
    # convert to RGBA
    
    c = mcolors.to_rgb(c) if type(c) == str else c
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gains = gain(np.linspace(0, 1, len(x)))
    alphas = alpha_min + (alpha_max - alpha_min) * gains

    ls = largs.pop('ls', '-')
    cmap = ListedColormap([(c, alpha) for alpha in alphas])
    if ls != 'none':
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        color_val = np.linspace(0, 1, len(x))
        # Set the values used for colormapping
        lc.set_array(color_val)
        lc.set_linewidth(lw)
        lc.set_linestyle(ls)
        line = ax.add_collection(lc, autolim=True)
        ax.autoscale_view()
    else:
        line = None

    # make markers at start and end
    m_args = dict(marker='o', zorder=10, ls='none')
    if type(mark) == int:
        # mark every n-th point
        n = len(x) // mark
        ax.plot(x[::n], y[::n], c=c, alpha=alpha_max, **(m_args | largs))
    else:
        if 'start' in mark.lower():
            ax.plot(x[0], y[0], c=c, alpha=alpha_min, **(m_args | largs))
        if 'end' in mark.lower():
            ax.plot(x[-1], y[-1], c=c, alpha=alpha_max,  **(m_args | largs))

        if 'all' in mark.lower():
            for i in range(len(x)):
                ax.plot(x[i], y[i], c=c, alpha=alphas[i], **(m_args | largs))

    return line,

def make_controls_fig(tape, cfg, fig=None, W_teachers=None, SVD_measures=False):
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    skip = int(max(cfg.T_tot // cfg.T_tape, 1))
    # skip = max(cfg.num_blocks * cfg.block_duration // tape.t.shape[-1], 1)

    if fig is None:
        TEXTWIDTH = 7.3  # inches
        TEXTHEIGHT = 10.5  # inches
        fig = plt.figure(figsize=(TEXTWIDTH/3*2, TEXTHEIGHT / 2.), layout="constrained")

    axd = fig.subplot_mosaic('''
    a
    b
    c
    d
    e
    f
    g
    h
    ''')

    fig.axd = axd

    # unshare x for 0 and a
    for k, v in axd.items():
        if k != "0":
            v.sharex(axd["a"])
            # v.sharey(axd["0"])

    if W_teachers is not None:
        try:
            plot_embed(axd["0"], tape, cfg, W_teachers)
        except ValueError:
            pass


    from lcs.plotting_utils import fill_between
    from functools import partial
    fill_between = partial(fill_between, gauss_reduce=True, line=True)
    t = tape.t.mean(0)
    print(t.min(), t.max())

    # %%
    ax = axd["a"]
    # window = cfg.block_duration*1//skip
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    fill_between(ax, t, tape.loss, label='loss',color="k", alpha=.5)

    ax.set_ylabel(r"$\mathcal{L}$")

    for k, ax in axd.items():
        for i, t_switch in enumerate(np.arange(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
            switch_point = int(0.75*cfg.num_blocks)
            if cfg.shared_teachers or cfg.shared_concat_teachers:
                if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B__AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = 'C2'
                elif cfg.c_gt_curriculum == 'AB_BC__':
                    c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_AB__':
                    c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = f'C{(i % cfg.num_contexts)+3}'
                elif cfg.c_gt_curriculum == 'AB_BC_CA__A_B_C__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)+cfg.num_contexts}'
                    else:
                        c = f'C{(i % cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                    c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded solution (not flexible)
                    if i < switch_point:
                        if i % 2 == 0:
                            c = 'C4'
                        else:
                            c = 'C6'
                    else:
                        c = 'C7'
                elif cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': # hard-coded solution
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + cfg.num_contexts}'
                    else:
                        c = f'C{(i % 2) + 2*cfg.num_contexts}'
            if cfg.c_gt_curriculum == 'A_B__':
                c = f'C{i % cfg.num_contexts}'
            ax.axvspan(t_switch, t_switch+cfg.block_duration, color=c, alpha=0.1, zorder=-10)

    # %%
    # norm of W
    ax = axd["b"]
    if (cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono') and cfg.context_model == False:
       fill_between(ax, t, tape.norm_W1.mean((-1)), label='norm of W1', c="k")
       if cfg.control == 'deep_mono':
           fill_between(ax, t, tape.norm_W2.mean((-1)), label='norm of W2', c="red")
    else:
        fill_between(ax, t, tape.norm_W1.mean((-1, -2)), label='norm of W', c="k") #TODO: make flexible for layer sizes
    ax.set_ylabel(r'$|W^p|$', color='k')

    # ax_tw = ax.twinx()
    # ax_tw.set_ylabel(r'$\bar \eta_c$', color='gray', rotation='horizontal', ha='left', va="center")

    ax.dataLim.y0 = 0
    # ax_tw.dataLim.y0 = 0
    ax.autoscale_view()
    # ax_tw.autoscale_view()

    limpad = .1

    # %%
    # norm of c
    ax = axd["c"]
    # ax.plot(t, tape.norm_c, label='norm of c', c="gray")
    ax.set_ylabel("sorted $c^p$")

    if cfg.control == 'N_diag_mono':
        for p in range(cfg.num_paths):
            fill_between(ax, t, np.mean(tape.sorted_c_student[:,:,p],axis=(2)), color=f'C{p}')
    elif cfg.control == 'deep_mono': # 10 x 1000 x 10 x 20
        for p in range(cfg.num_paths):
            for column in range(tape.sorted_W2_student.shape[4]):
                fill_between(ax, t, np.mean(tape.sorted_W2_student[:,:,p,:,column],axis=(2)), color=f'C{p}')
    else:
        for p in range(cfg.num_paths):
            fill_between(ax, t, tape.c1[..., p], color=f'C{p}', alpha=.5)

    if cfg.control == 'N_diag_mono' or cfg.control == '2_diag_mono':
        c_min = np.min(tape.c1)
        c_max = np.max(tape.c1)
    if cfg.control == 'deep_mono':
        c_min = np.min(tape.sorted_W2_student)
        c_max = np.max(tape.sorted_W2_student)
    ax.dataLim.y0 = c_min - limpad #-1 - limpad #0 - limpad
    ax.dataLim.y1 = c_max + limpad #1.5 + limpad
    ax.autoscale_view()

    # teacher 0, student 1
    pad = .02

    ax = axd["d"]

    if cfg.control == 'N_diag_mono' or cfg.control=='2_diag_mono' or cfg.control=='deep_mono':
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                if SVD_measures:
                    l, fill = fill_between(ax, t, tape.sorted_SVD_sim[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                else:
                    l, fill = fill_between(ax, t, tape.sorted_cossim[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                l.set_gapcolor(f'C{c}')
    else:
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                if SVD_measures:
                    l, fill = fill_between(ax, t, tape.SVD_similarity[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                else:
                    l, fill = fill_between(ax, t, tape.cos_sim1[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                l.set_gapcolor(f'C{c}')

    axd["d"].dataLim.y0 = 0 - limpad
    axd["d"].dataLim.y1 = 1 + limpad

    if cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control=='deep_mono':
        if SVD_measures:
            axd["d"].set_ylabel("sorted_SVD_sim")
        else:
            axd["d"].set_ylabel("sorted_cossim")
    else:
        if SVD_measures:
            axd["d"].set_ylabel("SVD_similarity")
        else:
            axd["d"].set_ylabel("cossim")

    ax = axd["e"] # C* alignment

    if cfg.control == 'N_diag_mono' or cfg.control=='2_diag_mono' or cfg.control=='deep_mono':
        if cfg.c_gt_curriculum == 'AB_BC_CA__' or cfg.c_gt_curriculum == 'AB_CA__AD__':
            pass
        else:
            for c in range(tape.c_alignment.shape[2]):
                if SVD_measures:
                    fill_between(ax, t, tape.SVD_c_alignment[:,:,c], color=f'C{c}', alpha=1) 
                else:
                    fill_between(ax, t, tape.c_alignment[:,:,c], color=f'C{c}', alpha=1) 

    axd["e"].dataLim.y0 = 0 - limpad
    axd["e"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["e"].set_ylabel("SVD_C*_alignment")
    else:
        axd["e"].set_ylabel("C*_alignment")


    ax = axd["f"] # concat cossim

    if cfg.control == 'N_diag_mono' or cfg.control=='2_diag_mono' or cfg.control=='deep_mono':
        if SVD_measures:
            fill_between(ax, t, tape.SVD_concat_cossim, alpha=1) 
        else:
            fill_between(ax, t, tape.concat_cossim, alpha=1) 

    axd["f"].dataLim.y0 = 0 - limpad
    axd["f"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["f"].set_ylabel("SVD_concat_cossim")
    else:
        axd["f"].set_ylabel("concat_cossim")


    if cfg.control == 'deep_mono':
         delta_W2 = np.linalg.norm(np.gradient(tape.W2, axis=1), axis=-1).mean((-1)) 
    else:
        delta_c =  np.gradient(tape.c1[...,-1], axis=1) #TODO: make flexible for layer sizes
    # diff along time, then vector norm, then mean over paths and features
    if (cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono' or cfg.control == 'deep_mono') and cfg.context_model == False:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-1)) #TODO: make flexible for layer sizes
    else:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-2, -1)) #TODO: make flexible for layer sizes
    # delta_W_mono = np.linalg.norm(np.gradient(tape.W_mono, axis=1), axis=-1).mean(-1)

    # make moving average with window n_repeats
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    if cfg.control == 'deep_mono':
        delta_W2 = maximum_filter1d(delta_W2, size=window, mode='nearest', axis=1)
    else:
        delta_c = maximum_filter1d(delta_c, size=window, mode='nearest', axis=1)
    delta_W = maximum_filter1d(delta_W, size=window, mode='nearest', axis=1)

    # axd["e"].set_yscale("log")
    ax = axd["g"]
    if cfg.control == 'deep_mono':
        fill_between(ax, t, delta_W2, color='red', ls="--")
    else:
        ax_tw = axd["g"].twinx()
        fill_between(ax_tw, t, delta_c, color='gray', ls="--")
        ax_tw.dataLim.y0 = 0
    # axd["e"].set_yscale("log")

    # ax_tw.set_yscale("log")
    fill_between(ax, t, delta_W, color='k', ls="-")
    # fill_between(ax_tw, t, delta_W_mono, color='tab:red', ls="-", zorder=-10)

    ax.dataLim.y0 = 0.
    ax.autoscale_view()

    # ylable
    axd["g"].set_ylabel(r"$\langle |\Delta W| \rangle$", color='k')
    if cfg.control != 'deep_mono':
        ax_tw.set_ylabel(r"$\langle \Delta c \rangle$", color='gray', rotation='horizontal', ha='left', va="center")

    axd["g"].set_xlabel("time $t$")
    if cfg.control != 'deep_mono':
        ax_tw.set_zorder(axd["h"].get_zorder()+1)

    ax = axd["h"]
    if SVD_measures:
        l, fill = fill_between(ax, t, tape.SVD_alignment, color='k', label="context alignment", alpha=1)
    else:
        l, fill = fill_between(ax, t, tape.context_alignment1, color='k', label="context alignment", alpha=1)
    l.set_gapcolor("C0")
    axd["h"].dataLim.y0 = 0 - limpad
    axd["h"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["h"].set_ylabel("SVD Alignment")
    else:
        axd["h"].set_ylabel("Alignment")

    # some polishing
    # labels = [r"$\mathbf{D_1}$", r"$\mathbf{D_2}$", r"$\mathbf{D_3}$", r"$\mathbf{D_4}$", r"$\mathbf{D_5}$", r"$\mathbf{D_6}$", r"$\mathbf{D_7}$", r"$\mathbf{D_8}$", r"$\mathbf{D_9}$"] #[r"$\mathbf{D_0}$",r"$\mathbf{D_1}$", r"$\mathbf{D_2}$", r"$\mathbf{D_3}$", r"$\mathbf{D_4}$", r"$\mathbf{D_5}$", r"$\mathbf{D_6}$"]
    for i, ax in enumerate(axd.values()):
        # ax.text(x=-.15, y=1.2, s=labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.yaxis.label.set(rotation='horizontal', ha='right', va="center")

    fig.align_labels()

    return fig

@contextmanager
def no_autoscale(ax=None, axis="both"):
    ax = ax or plt.gca()
    ax.figure.canvas.draw()
    lims = [ax.get_xlim(), ax.get_ylim()]
    yield
    if axis == "both" or axis == "x":
        ax.set_xlim(*lims[0])
    if axis == "both" or axis == "y":
        ax.set_ylim(*lims[1])


def plot_embed(ax, tape, cfg, W_teachers):
    # Create the animation
    ax.set_box_aspect(1)

    # grey circle in the background on ax1
    circle = plt.Circle((0, 0), 1, color='grey', alpha=0.1)

    # make t, p and i batch dimensions
    n_out = cfg.layer_sizes[-1]
    n_paths = cfg.num_paths
    n_contexts = cfg.num_contexts
    N_trials = tape.t.shape[0]
    n_times = tape.t.shape[1]

    all_tranforms = np.empty((N_trials, n_out), dtype=object)
    for i_tr in range(N_trials):
        for i in range(n_out):
            embed = TruncatedSVD(n_components=2)  # to avoid centering
            W_teacher_fit = W_teachers[i_tr, :, i].copy()
            W_teacher_fit[0] *= (1.001) # hack to align it with the x-axis
            embed.fit(W_teacher_fit)
            all_tranforms[i_tr, i] = (embed.transform)

    
    W_student_embed = np.zeros((N_trials, n_times, n_paths, n_out, 2))
    l = cfg.num_layers
    for i_tr in range(N_trials):
        for p in range(n_paths):
            for i in range(n_out):
                transform = all_tranforms[i_tr, i]
                W_student_embed[i_tr, :,p,i] = transform(tape[f'W{l}'][i_tr, :, p, i, :])

    # smoothen with a moving average
    W_student_embed = gaussian_filter1d(W_student_embed, sigma=max(len(W_student_embed) // 10, 1), axis=1, mode='nearest')
    c_embed = maximum_filter1d(tape[f'c{l}'], size=max((n_times) // 10, 1), axis=1, mode='nearest')

    # define a symlog trafo
    def symlog(x, linthresh=1e-2):
        a = 1.0
        return np.sign(x)*np.abs(x)**a


    # Define the plot function
    def plot_embed(ti):
        # Clear the plot
        ax.cla()
        ax.axis('off')
        ax.add_artist(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        # ax1.set_yscale("symlog", linthresh=1e-2)
        # ax1.set_xscale("symlog", linthresh=1e-2)
        
        # Plot the teacher weights
        for i_tr in range(N_trials):
            for p in range(cfg.num_contexts):
                # pij'
                for i in range(n_out):
                    transform = all_tranforms[i_tr, i]
                    W_teacher_embed = transform(np.atleast_2d(W_teachers[i_tr, p, i]))
                    # W_teacher_embed = W_teacher_embed / np.linalg.norm(W_teacher_embed)
                    ax.scatter(W_teacher_embed[:, 0], W_teacher_embed[:, 1], s=100, marker='x', color='k', zorder=10)
                    ax.plot(W_teacher_embed[:, 0], W_teacher_embed[:, 1], color=plt.cm.tab10(p), alpha=0.5, zorder=10)
        
        # Plot the student weights
        tis_show = np.arange(ti)

        for i_tr in range(N_trials):
            for p in range(n_paths):
                c_ = c_embed[i_tr, tis_show, p]
                # c_ = 1
                for i in range(n_out):
                    if i > 0: break
                    # W_student_embed = W_student_embed / np.linalg.norm(W_student_embed, axis=-1, keepdims=True)
                    colors = [(*plt.cm.tab10(p)[:3], (1-np.exp(-.1*i/len(W_student_embed[i_tr, tis_show])))) for i in range(len(W_student_embed[i_tr, tis_show]))]
                    
                    # print(W_student_embed.shape)
                    ax.scatter(symlog(c_*W_student_embed[i_tr, tis_show, p, i, 0]), symlog(c_*W_student_embed[i_tr, tis_show, p, i, 1]), s=10, c=colors)
                    ax.plot(symlog(c_*W_student_embed[i_tr, tis_show, p, i, 0]), symlog(c_*W_student_embed[i_tr, tis_show, p, i, 1]), color=plt.cm.tab10(p), alpha=0.5)

        
        # Add a title

    plot_embed(n_times)


def format_legend(fig):
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right', frameon=False)



def get_SVD_transforms(W_teacher): 
    """
    W_teacher: pij
    """
    transforms = []
    for i in range(W_teacher.shape[1]):
        embed = TruncatedSVD(n_components=2)
        W_teacher_fit = W_teacher[:,i].copy()
        W_teacher_fit[0] *= (1.001)
        embed.fit(W_teacher_fit)
        transforms.append(embed.transform)
    return transforms

def symlog(x, linthresh=1e-2):
    a = 1.0
    return np.sign(x)*np.abs(x)**a

def flatten_timesteps(data):
    shape_dim = (data.shape[0], data.shape[1]*data.shape[2])
    for i in range(3, len(data.shape)):
        shape_dim += (data.shape[i],)
    return data.reshape(shape_dim)

def plot_sv_dynamics(data_folder, seed): # only done for a single seed; NOT TESTED
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    ax1.set_box_aspect(1)
    circle = plt.Circle((0, 0), 1, color='grey', alpha=0.1)
    
    W = np.load(os.path.join(data_folder, 'W.npy'))
    W = flatten_timesteps(W)[seed]
    c = np.load(os.path.join(data_folder, 'c.npy'))
    c = flatten_timesteps(c)[seed]

    W_teacher = np.load(os.path.join(data_folder, 'W_teachers.npy'))[seed]
    transforms = get_SVD_transforms(W_teacher)

    W_embed = np.zeros((W.shape[0], W.shape[1], W.shape[2], 2))  # time t, path p, output i, embedding dim
    for p in range(W.shape[1]):
        for i in range(W.shape[2]):
            transform = transforms[i]
            W_embed[:,p,i] = np.array([transform(W[:,p,i,:])])  # time t, path p, output i, input j 

    W_embed = gaussian_filter1d(W_embed, sigma=len(W_embed) // 10, axis=0, mode='nearest')
    c_embed = maximum_filter1d(c, size=len(c) // 10, axis=0, mode='nearest')

    def plot_embed(ti):
        ax1.cla()
        ax2.cla()
        ax1.axis('off')
        ax1.add_artist(circle)
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax2.set_xlim(0, W.shape[0])
        
        for p in range(W_teacher.shape[0]):
            for i in range(W_teacher.shape[1]):
                transform = transforms[i]
                W_teacher_embed = transform(np.atleast_2d(W_teacher[p, i]))
                ax1.scatter(W_teacher_embed[:, 0], W_teacher_embed[:, 1], s=100, marker='x', color='k', zorder=10)
                ax1.plot(W_teacher_embed[:, 0], W_teacher_embed[:, 1], color=plt.cm.tab10(p), alpha=0.5, zorder=10)
    
        tis_show = np.arange(ti)
        ts_show = np.arange(ti)
    
        for p in range(W_embed.shape[1]):
            c_ = c_embed[tis_show, p]
            for i in range(W_embed.shape[2]):
                if i > 0: break
                colors = [(*plt.cm.tab10(p)[:3], (1-np.exp(-.1*i/len(W_embed[tis_show])))) for i in range(len(W_embed[tis_show]))]
                ax1.scatter(symlog(c_*W_embed[tis_show, p, i, 0]), symlog(c_*W_embed[tis_show, p, i, 1]), s=10, c=colors)
                ax1.plot(symlog(c_*W_embed[tis_show, p, i, 0]), symlog(c_*W_embed[tis_show, p, i, 1]), color=plt.cm.tab10(p), alpha=0.5)
    
        ax2.plot(ts_show, c[tis_show, 0], c=plt.cm.tab10(0), alpha=0.5)
        ax2.plot(ts_show, c[tis_show, 1], c=plt.cm.tab10(1), alpha=0.5)
        
        ax2.set_ylim(.5 - 1, .5+1)
    
        ax1.set_title(f'MDS Embedding of W_student Dynamics at time {ti}')

    frames = np.linspace(1, W.shape[0], 10).astype(int)
    ani = animation.FuncAnimation(fig, plot_embed, frames=frames, interval=50)

    plot_embed(frames[-1])

    HTML(ani.to_jshtml())


def plot_cs(tape, cfg, sorted=True):
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    TEXTWIDTH = 7.3  # inches
    TEXTHEIGHT = 10.5  # inches
    fig = plt.figure(figsize=(TEXTWIDTH/3*2, TEXTHEIGHT / 2.), layout="constrained")
    
    axd = fig.subplot_mosaic('''
    a
    ''')

    fig.axd = axd

    # unshare x for 0 and a
    for k, v in axd.items():
        if k != "0":
            v.sharex(axd["a"])
            # v.sharey(axd["0"])

    from lcs.plotting_utils import fill_between
    from functools import partial
    fill_between = partial(fill_between, gauss_reduce=True, line=True)
    t = tape.t.mean(0)
    print(t.min(), t.max())

    # %%
    ax = axd["a"]

    for k, ax in axd.items():
        if k not in "0e":
            for i, t_switch in enumerate(np.arange(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
                switch_point = int(0.75*cfg.num_blocks)
                if cfg.shared_teachers or cfg.shared_concat_teachers:
                    if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                    elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B__AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = 'C2'
                    elif cfg.c_gt_curriculum == 'AB_BC__':
                        c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_AB__':
                        c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = f'C{(i % cfg.num_contexts)+3}'
                    elif cfg.c_gt_curriculum == 'AB_BC_CA__A_B_C__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)+cfg.num_contexts}'
                        else:
                            c = f'C{(i % cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                        c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded solution (not flexible)
                        if i < switch_point:
                            if i % 2 == 0:
                                c = 'C4'
                            else:
                                c = 'C6'
                        else:
                            c = 'C7'
                    elif cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': # hard-coded solution
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + cfg.num_contexts}'
                        else:
                            c = f'C{(i % 2) + 2*cfg.num_contexts}'

                if cfg.c_gt_curriculum == 'A_B__':
                    c = f'C{i % cfg.num_contexts}'
                ax.axvspan(t_switch, t_switch+cfg.block_duration, color=c, alpha=0.1, zorder=-10)

    limpad = .1

    ax.set_ylabel("$c^p$")
    p=-1
    if cfg.control == 'N_diag_mono':
        if sorted:
            for p in range(cfg.num_paths):
                fill_between(ax, t, np.mean(tape.sorted_c_student[:,:,p],axis=(2)), color=f'C{p}')
        else:
            for p in range(cfg.num_paths*cfg.hidden_size):
                fill_between(ax, t, tape.c1[..., p], color=f'C{p}', alpha=.5)
    elif cfg.control == 'deep_mono': 
        if sorted:
            for p in range(cfg.num_paths):
                fill_between(ax, t, np.mean(tape.sorted_W2_student[:,:,p],axis=(2,3)), color=f'C{p}')
        else:
            for col in range(tape.W2.shape[3]): # mean across rows within column
                fill_between(ax, t, np.mean(tape.W2[:,:,:,col],axis=2), color=f'C{col}', alpha=.5)
    else:
        for p in range(cfg.num_paths):
            fill_between(ax, t, tape.c1[..., p], color=f'C{p}', alpha=.5)

    if cfg.control=='deep_mono':
        c_min = np.min(tape.W2)
        c_max = np.max(tape.W2)
    else:
        c_min = np.min(tape.c1)
        c_max = np.max(tape.c1)
        
    ax.dataLim.y0 = c_min - limpad #-1 - limpad #0 - limpad
    ax.dataLim.y1 = c_max + limpad #1.5 + limpad
    ax.autoscale_view()

    return fig

def plot_loss_at_switch(tape, cfg, alignment=False):
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    TEXTWIDTH = 7.3  # inches
    TEXTHEIGHT = 10.5  # inches
    fig = plt.figure(figsize=(TEXTWIDTH/3*2, TEXTHEIGHT / 2.), layout="constrained")
    ax = plt.gca()

    from lcs.plotting_utils import fill_between
    from functools import partial
    fill_between = partial(fill_between, gauss_reduce=True, line=True)
    t = tape.t.mean(0)
    print(t.min(), t.max())

    # %%
    colormap = plt.get_cmap('jet')
    ax.set_prop_cycle(matplotlib.cycler(color=[colormap(k) for k in np.linspace(0, 1, cfg.num_blocks)]) )
    block = 0
    for t in range(tape.current_context[0].shape[0]-1):
        if t + 10 < tape.t.shape[1]:
            if tape.current_context[0][t] != tape.current_context[0][t+1]:
                if block % 10 == 0:
                    if alignment:
                        fill_between(ax, np.linspace(-5,10, 15), tape.context_alignment1[:, t-5:t+10], label=f'switch {block}', alpha=.5) 
                    else:
                        fill_between(ax, np.linspace(-5,10, 15), tape.loss[:, t-5:t+10], label=f'switch {block}', alpha=.5) 
                else:
                    if alignment:
                        fill_between(ax, np.linspace(-5,10, 15), tape.context_alignment1[:, t-5:t+10], alpha=.5) 
                    else:
                        fill_between(ax, np.linspace(-5,10,15), tape.loss[:, t-5:t+10], alpha=.5) 
                        # fill_between(ax, np.linspace(t1,t2, 15), tape.loss[:, t-5:t+10], alpha=.5) 
                block += 1
    if alignment:
        ax.set_ylabel("Alignment")
    else:
        ax.set_ylabel(r"$\mathcal{L}$")
    ax.autoscale_view()
    ax.legend()
    return fig

def plot_sorted_cossim_heatmap(tape, cfg):
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    mean_row_cossim = np.mean(tape.row_cossim, axis=1)

    last_c_idx_1 = np.where(tape.current_context[0] == 0)[0][-1]
    last_c_idx_2 = np.where(tape.current_context[0] == 1)[0][-1]
    W2_dif_1 = tape.sorted_W2_student[0, last_c_idx_1][0] - tape.sorted_W2_student[0, last_c_idx_2][0]
    W2_dif_2 = tape.sorted_W2_student[0, last_c_idx_1][1] - tape.sorted_W2_student[0, last_c_idx_2][1]
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    ax1, ax2 = axes

    im1 = ax1.matshow(W2_dif_1, cmap='seismic')
    im2 = ax2.matshow(W2_dif_2, cmap='seismic')

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)   
    return fig


def make_LCS_fig(tape, cfg, fig=None, W_teachers=None, SVD_measures=False):
    plt = init_mpl()
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    matplotlib.rcParams['axes.spines.top'] = False

    skip = int(max(cfg.T_tot // cfg.T_tape, 1))
    # skip = max(cfg.num_blocks * cfg.block_duration // tape.t.shape[-1], 1)

    if fig is None:
        TEXTWIDTH = 7.3  # inches
        TEXTHEIGHT = 10.5  # inches
        fig = plt.figure(figsize=(TEXTWIDTH/3*2, TEXTHEIGHT / 2.), layout="constrained")

    axd = fig.subplot_mosaic('''
    a
    b
    c
    d
    e
    f
    g
    ''')

    fig.axd = axd

    # unshare x for 0 and a
    for k, v in axd.items():
        if k != "0":
            v.sharex(axd["a"])


    if W_teachers is not None:
        try:
            plot_embed(axd["0"], tape, cfg, W_teachers)
        except ValueError:
            pass


    from lcs.plotting_utils import fill_between
    from functools import partial
    fill_between = partial(fill_between, gauss_reduce=True, line=True)
    t = tape.t.mean(0)
    print(t.min(), t.max())

    # %%
    ax = axd["a"]
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))
    # window = cfg.block_duration*1//skip

    fill_between(ax, t, tape.loss, label='loss',color="k", alpha=.5)

    ax.set_ylabel(r"$\mathcal{L}$")

    for k, ax in axd.items():
        if k not in "0e":
            for i, t_switch in enumerate(range(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
                switch_point = int(0.75*cfg.num_blocks)
                if cfg.shared_teachers or cfg.shared_concat_teachers:
                    if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                    elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + 1}'
                        else:
                            c = f'C{(i % cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B__AB__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = 'C2'
                    elif cfg.c_gt_curriculum == 'AB_BC__':
                        c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_AB__':
                        c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                    elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)}'
                        else:
                            c = f'C{(i % cfg.num_contexts)+cfg.num_contexts}'
                    elif cfg.c_gt_curriculum == 'AB_BC_CA__A_B_C__':
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts)+cfg.num_contexts}'
                        else:
                            c = f'C{(i % cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                        c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                    elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded solution (not flexible)
                        if i < switch_point:
                            if i % 2 == 0:
                                c = 'C4'
                            else:
                                c = 'C6'
                        else:
                            c = 'C7'
                    elif cfg.c_gt_curriculum == 'AB_BC_CD_DA__AC_BD__': # hard-coded solution
                        if i < switch_point:
                            c = f'C{(i % cfg.num_contexts) + cfg.num_contexts}'
                        else:
                            c = f'C{(i % 2) + 2*cfg.num_contexts}'
                if cfg.c_gt_curriculum == 'A_B__':
                    c = f'C{i % cfg.num_contexts}'
                ax.axvspan(t_switch, t_switch+cfg.block_duration, color=c, alpha=0.1, zorder=-10)

    # %%
    # norm of W
    ax = axd["b"]
    if (cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono') and cfg.context_model == False:
       fill_between(ax, t, tape.norm_W1.mean((-1)), label='norm of W', c="k")
    else:
        fill_between(ax, t, tape.norm_W1.mean((-1, -2)), label='norm of W', c="k") #TODO: make flexible for layer sizes
    ax.set_ylabel(r'$|W^p|$', color='k')

    ax_tw = ax.twinx()
    ax_tw.set_ylabel(r'$\bar \eta_c$', color='gray', rotation='horizontal', ha='left', va="center")

    ax.dataLim.y0 = 0
    ax_tw.dataLim.y0 = 0
    ax.autoscale_view()
    ax_tw.autoscale_view()

    limpad = .1

    # %%
    # norm of c
    ax = axd["c"]
    ax.set_ylabel("$c^p$")

    if cfg.control == 'N_diag_mono':
        for p in range(cfg.num_paths*cfg.hidden_size):
            fill_between(ax, t, tape.c1[..., p], color=f'C{p}', alpha=.5)
    else:
        for p in range(cfg.num_paths):
            fill_between(ax, t, tape.c1[..., p], color=f'C{p}', alpha=.5)
    # ax.set_yticks([0, .5, 1])
    c_min = np.min(tape.c1)
    ax.dataLim.y0 = c_min - limpad #-1 - limpad #0 - limpad
    c_max = np.max(tape.c1)
    ax.dataLim.y1 = c_max + limpad #1.5 + limpad
    ax.autoscale_view()

    # teacher 0, student 1
    pad = .02

    ax = axd["d"]

    if cfg.control == 'N_diag_mono' or cfg.control=='2_diag_mono':
        # for c in range(cfg.num_contexts):
        #     fill_between(ax, t, tape.c_alignment[:,:,c], color=f'C{c}', alpha=1) 
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                l, fill = fill_between(ax, t, tape.sorted_cossim[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                l.set_gapcolor(f'C{c}')
    else:
        for p in range(cfg.num_paths):
            for c in range(cfg.num_contexts):
                if p == c:
                    t_pad = (-2**p)*pad
                else:
                    t_pad = 0
                if SVD_measures:
                    l, fill = fill_between(ax, t, tape.SVD_similarity[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                else:
                    l, fill = fill_between(ax, t, tape.cos_sim1[..., c, p]+t_pad, color=f'C{p}', label="cosine similarity", alpha=1, ls="dashed")
                l.set_gapcolor(f'C{c}')

    axd["d"].dataLim.y0 = 0 - limpad
    axd["d"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["d"].set_ylabel("SVD sim")
    else:
        axd["d"].set_ylabel(r"$\frac{W^p \cdot W^{p'}}{|W^p| \cdot |W^{p'}|}$")


    delta_c =  np.gradient(tape.c1[...,-1], axis=1) #TODO: make flexible for layer sizes
    # diff along time, then vector norm, then mean over paths and features
    if (cfg.control == '2_diag_mono' or cfg.control == 'N_diag_mono') and cfg.context_model == False:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-1)) #TODO: make flexible for layer sizes
    else:
        delta_W = np.linalg.norm(np.gradient(tape.W1, axis=1), axis=-1).mean((-2, -1)) #TODO: make flexible for layer sizes
    # delta_W_mono = np.linalg.norm(np.gradient(tape.W_mono, axis=1), axis=-1).mean(-1)

    # make moving average with window n_repeats
    # window = max((cfg.block_length * 1 // skip) * 2, 1)
    window = int(max((cfg.block_duration * 1 // skip) * 2, 1))

    delta_c = maximum_filter1d(delta_c, size=window, mode='nearest', axis=1)
    delta_W = maximum_filter1d(delta_W, size=window, mode='nearest', axis=1)

    ax = axd["e"]
    ax_tw = axd["e"].twinx()
    fill_between(ax_tw, t, delta_c, color='gray', ls="--")
    # axd["e"].set_yscale("log")

    # ax_tw.set_yscale("log")
    fill_between(ax, t, delta_W, color='k', ls="-")
    # fill_between(ax_tw, t, delta_W_mono, color='tab:red', ls="-", zorder=-10)

    ax_tw.dataLim.y0 = 0
    ax.dataLim.y0 = 0.
    ax.autoscale_view()

    # ylable
    axd["e"].set_ylabel(r"$\langle |\Delta W| \rangle$", color='k')
    ax_tw.set_ylabel(r"$\langle \Delta c \rangle$", color='gray', rotation='horizontal', ha='left', va="center")

    ax_tw.set_zorder(axd["e"].get_zorder()+1)

    ax = axd["f"]
    if SVD_measures:
        l, fill = fill_between(ax, t, tape.SVD_alignment, color='k', label="context alignment", alpha=1)
    else:
        l, fill = fill_between(ax, t, tape.context_alignment1, color='k', label="context alignment", alpha=1)
    l.set_gapcolor("C0")
    axd["f"].dataLim.y0 = 0 - limpad
    axd["f"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["f"].set_ylabel("SVD Alignment")
    else:
        axd["f"].set_ylabel("Alignment")
    axd["f"].set_xlabel("time $t$")

    ax = axd["g"] # concat cossim

    if SVD_measures:
        fill_between(ax, t, tape.concat_SVD, alpha=1) 
    else:
        fill_between(ax, t, tape.concat_cossim, alpha=1) 

    axd["g"].dataLim.y0 = 0 - limpad
    axd["g"].dataLim.y1 = 1 + limpad

    if SVD_measures:
        axd["g"].set_ylabel("concat_SVD")
    else:
        axd["g"].set_ylabel("concat_cossim")



    # some polishing
    labels = [r"$\mathbf{D_1}$", r"$\mathbf{D_2}$", r"$\mathbf{D_3}$", r"$\mathbf{D_4}$", r"$\mathbf{D_5}$", r"$\mathbf{D_6}$", r"$\mathbf{D_7}$"] #[r"$\mathbf{D_0}$",r"$\mathbf{D_1}$", r"$\mathbf{D_2}$", r"$\mathbf{D_3}$", r"$\mathbf{D_4}$", r"$\mathbf{D_5}$", r"$\mathbf{D_6}$"]
    for i, ax in enumerate(axd.values()):
        ax.text(x=-.15, y=1.2, s=labels[i], transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='right')
        ax.yaxis.label.set(rotation='horizontal', ha='right', va="center")

    fig.align_labels()

    return fig


def plot_concat_similarity(ax, tape, cfg, sim=None, **largs):
    
    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    pad = .02
    t = np.atleast_1d(tape.t).mean(0)

    fill_between(ax, t, tape.concat_cossim, **(dict(color="k", alpha=1.) | largs)) 

    ax.dataLim.y0 = 0 - limpad
    ax.dataLim.y1 = 1 + limpad

    ax.set_ylabel(r"$\frac{W^p \cdot W^{p'}}{|W^p| \cdot |W^{p'}|}$")

def indicate_contexts_old(ax, tape, cfg, **largs):
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import seaborn as sns

    tape = jax.tree.map(lambda x: x[None], tape) if tape.t.ndim == 1 else tape
    colors = sns.color_palette("deep", as_cmap=True)
    cmap = LinearSegmentedColormap.from_list('ctx_colors', colors)

    t = tape.t.mean(0)
    if cfg.Y_tgt is None: 
        color_indcs = np.zeros_like(t).astype(int)
        
        for i, t_switch in enumerate(np.arange(0, cfg.block_duration*cfg.num_blocks, cfg.block_duration)):
            switch_point = int(0.75*cfg.num_blocks)
            if cfg.shared_teachers or cfg.shared_concat_teachers:
                if cfg.c_gt_curriculum == 'B_AB__A_B_AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % (cfg.num_contexts+cfg.num_shared_contexts))}'
                elif cfg.c_gt_curriculum == 'B_AB__A_B__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts) + 1}'
                    else:
                        c = f'C{(i % cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B__AB__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = 'C2'
                elif cfg.c_gt_curriculum == 'AB_BC__':
                    c = f'C{((i % (cfg.num_contexts-1)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_AB__':
                    c = f'C{i % (cfg.num_contexts + cfg.num_shared_contexts)}'
                elif cfg.c_gt_curriculum == 'A_B_C__AB_BC_CA__':
                    if i < switch_point:
                        c = f'C{(i % cfg.num_contexts)}'
                    else:
                        c = f'C{(i % cfg.num_contexts)+3}'
                elif cfg.c_gt_curriculum == 'AB_BC_CA__':
                    c = f'C{((i % (cfg.num_contexts)) + cfg.num_contexts)}'
                elif cfg.c_gt_curriculum == 'AB_CD__AD__': # hard-coded
                    if i < switch_point:
                        if i % 2 == 0:
                            c = 'C4'
                        else:
                            c = 'C6'
                    else:
                        c = 'C7'
            if cfg.c_gt_curriculum == 'A_B__':
                c = f'C{i % cfg.num_contexts}'
            ax.axvspan(t_switch-(cfg.dt*i), t_switch+cfg.block_duration-(cfg.dt*i), color=c, alpha=0.1, zorder=-10, rasterized=True)
            # color_indcs[(t >= t_switch) & (t < t_switch+cfg.block_duration)] = int(c[1:])

    else: 
        # make listed cmap
        colors = plt.cm.tab10(np.arange(cfg.num_contexts))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('ctx_colors', colors)
        Y_tgt_ = (cfg.Y_tgt(t)*np.arange(cfg.num_contexts)).sum(-1)
        with no_autoscale(ax):
            ax.pcolormesh(t, [-100, 100], np.repeat(Y_tgt_[:, None], 2, axis=1).T, cmap=cmap, alpha=0.1, zorder=-10, rasterized=True)
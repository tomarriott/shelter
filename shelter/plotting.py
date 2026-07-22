import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullLocator
from matplotlib.patheffects import Stroke, Normal, withStroke
from .utils import extract_kwargs
from .data import bin_data, fold_data, get_transits_in_data
from .io import get_directory, find_path
from .colour import Colour, Gradient

def use_custom_styles():
    plt.style.use(find_path('styles/light_style.mplstyle'))
    plt.style.use(find_path('styles/use_serif.mplstyle'))
    plt.style.use(find_path('styles/paper_text.mplstyle'))

def light_style():
    plt.style.use(find_path('styles/light_style.mplstyle'))

def dark_style():
    plt.style.use(find_path('styles/dark_style.mplstyle'))

def serif_style():
    plt.style.use(find_path('styles/use_serif.mplstyle'))

def sans_style():
    plt.style.use(find_path('styles/use_sans.mplstyle'))

def paper_style():
    plt.style.use(find_path('styles/paper_text.mplstyle'))

def talk_style():
    plt.style.use(find_path('styles/talk_text.mplstyle'))

################################################################################
# - PLOTTING DATA ------------------------------------------------------------ #
################################################################################

def plot_axes(func, *args, figsize=(10, 6), save=False, save_path='', **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()

    func(ax, *args)

    fig.tight_layout()

    if not save:
        plt.show()

    if save:
        savefig_args = extract_kwargs(plt.savefig, kwargs)
        plt.savefig(save_path, **savefig_args)


# Thanks Claude
def split_axis(ax, nrows=1, ncols=2, share='x', hspace=0.05, wspace=0.05, ratios=None):
    """
    Replace `ax` with a grid of sub-axes occupying the same position,
    sharing the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to split. Must belong to a figure using GridSpec-based subplots.
    nrows, ncols : int
        Shape of the sub-grid to create ßin place of `ax`.
    share : {'x', 'y', 'both', None}
        Which axis to share across the new sub-axes.
    hspace, wspace : float
        Spacing between the new sub-axes.
    ratios : dict, optional
        e.g. {'height_ratios': [3, 1]} or {'width_ratios': [2, 1]}

    Returns
    -------
    list of Axes, in row-major order
    """
    fig = ax.figure
    subplotspec = ax.get_subplotspec()
    #ax.remove()  # drop the original axis but keep its slot
    ax.set_axis_off()

    ratios = ratios or {}
    sub_gs = subplotspec.subgridspec(nrows, ncols, hspace=hspace, wspace=wspace, **ratios)

    axes = []
    ref_ax = None
    for i in range(nrows):
        for j in range(ncols):
            sharex = ref_ax if share in ('x', 'both') else None
            sharey = ref_ax if share in ('y', 'both') else None
            new_ax = fig.add_subplot(sub_gs[i, j], sharex=sharex, sharey=sharey)
            if ref_ax is None:
                ref_ax = new_ax
            axes.append(new_ax)

    return axes


def residual_line(ax, median, uncertainty, nlines=3, colour1='#40A1A1', colour2='k', colour3='w'):
    ax.axhline(median, c=colour1, linestyle='-', zorder=10, path_effects=[Stroke(linewidth=3, foreground=colour3, alpha=0.5), Normal()])

    lines = ['--', '-.', ':']
    alphas = [0.5, 0.25, 0.1]

    for n in range(nlines):
        ax.axhline(median + ((n + 1) * uncertainty), c=colour2, linestyle=lines[n], alpha=alphas[n], zorder=10)
        ax.axhline(median - ((n + 1) * uncertainty), c=colour2, linestyle=lines[n], alpha=alphas[n], zorder=10)


# ---------------------------------------------------------------------------- #
# Lightcurve plotting                                                          #
# ---------------------------------------------------------------------------- #

def ax_lightcurve(ax, t, y, yerr=None, transit_times=[], plot_bin=False,
                  data_errorbar_args={'ms':1, 'ls':'none', 'c':'#f04f4f', 'fmt':'o', 'mfc':'#f04f4f', 'mec':'#4f2020', 'alpha':0.5, 'zorder':2,},
                  bin_data_args={},
                  bin_errorbar_args={},
                  **kwargs):
    if ax is None:
        ax = plt.axes()

    if yerr is None:
        yerr = np.zeros(np.shape(y))
    
    ax.errorbar(t, y, yerr=yerr, **data_errorbar_args)

    if isinstance(transit_times, dict):
        transit_times = transit_times.values()
    if isinstance(transit_times, list):
        transit_times = np.array(transit_times)

    for transit in transit_times:
        ax.axvline(transit, c='#40a1a1', alpha=0.5, zorder=0, linestyle='--')

    if plot_bin:
        t_bin, y_bin, yerr_bin = bin_data(t, y, yerr, **bin_data_args)

        ax.errorbar(t_bin, y_bin, yerr_bin, ms=4, capsize=2, elinewidth=1, fmt='o', mfc='w', mec='k', ecolor='k', zorder=20, **bin_errorbar_args)

    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Flux')


def plot_lightcurve(t, y, yerr=None, transit_times=[], plot_bin=False,
                    figsize=(10, 6), save=False, save_path=os.path.join(get_directory(), 'lightcurve.png'),
                    data_errorbar_args={'ms':1, 'ls':'none', 'c':'#f04f4f', 'fmt':'o', 'mfc':'#f04f4f', 'mec':'#4f2020', 'alpha':0.5, 'zorder':2,},
                    bin_data_args={},
                    bin_errorbar_args={},
                    **kwargs):
    plot_axes(ax_lightcurve, t, y, yerr, transit_times, plot_bin,
              figsize=figsize, save=save, save_path=save_path,
              data_errorbar_args=data_errorbar_args, bin_data_args=bin_data_args, bin_errorbar_args=bin_errorbar_args, **kwargs)


# ---------------------------------------------------------------------------- #
# Phasefold plotting                                                           #
# ---------------------------------------------------------------------------- #

def ax_phasefold(ax, t, y, yerr=None, period=None, t0=None, duration=None, depth=None, plot_bin=True,
                 data_errorbar_args={'ms':1, 'ls':'none', 'c':'#f04f4f', 'fmt':'o', 'mfc':'#f04f4f', 'mec':'#4f2020', 'alpha':0.5, 'zorder':2,},
                 bin_data_args={'n_points': 200},
                 bin_errorbar_args={},
                 **kwargs):
    ax_lightcurve(ax, *fold_data(t, y, period=period, t0=t0, e=yerr), plot_bin=plot_bin,
                  data_errorbar_args=data_errorbar_args, bin_data_args=bin_data_args, bin_errorbar_args=bin_errorbar_args, **kwargs)

    ax.set_xlabel('Phase')

    if (period is not None) and (duration is not None):
        xlims = -duration/(period), duration/(period)
        ax.set_xlim(xlims)
    if depth is not None:
        ylims = -3*depth + 1, 2*depth + 1
        ax.set_ylim(ylims)


def plot_phasefold(t, y, yerr=None, period=None, t0=None, duration=None, depth=None, plot_bin=True,
                   data_errorbar_args={'ms':1, 'ls':'none', 'c':'#f04f4f', 'fmt':'o', 'mfc':'#f04f4f', 'mec':'#4f2020', 'alpha':0.5, 'zorder':2,},
                   bin_data_args={'n_points': 200},
                   bin_errorbar_args={},
                   **kwargs):
    plot_axes(ax_phasefold, t, y, yerr, period, t0, duration, depth, plot_bin, bin_data_args, data_errorbar_args, bin_errorbar_args, **kwargs)


def ax_oddeven(ax, t, y, yerr=None, period=None, t0=None, plot_bin=True,
               data_errorbar_args={'ms':1, 'ls':'none', 'c':'#f04f4f', 'fmt':'o', 'mfc':'#f04f4f', 'mec':'#4f2020', 'alpha':0.5, 'zorder':2,},
               bin_data_args={'n_points': 200},
               bin_errorbar_args={},
               **kwargs):
    axes = split_axis(ax, nrows=1, ncols=2, share='none', hspace=0, wspace=0)

    ax_lightcurve(axes[0], *fold_data(t, y, period=2*period, t0=t0, e=yerr), plot_bin=plot_bin,
                  data_errorbar_args=data_errorbar_args, bin_data_args=bin_data_args, bin_errorbar_args=bin_errorbar_args, **kwargs)
    
    axes[0].set_xlabel('Phase')
    axes[0].set_ylabel('Flux')
    axes[0].yaxis.set_ticks_position('left')
    
    ax_lightcurve(axes[1], *fold_data(t, y, period=2*period, t0=(t0+period), e=yerr), plot_bin=plot_bin,
                  data_errorbar_args=data_errorbar_args, bin_data_args=bin_data_args, bin_errorbar_args=bin_errorbar_args, **kwargs)
    
    axes[1].set_xlabel('Phase')
    axes[1].set_ylabel('')
    axes[1].yaxis.set_ticks_position('right')
    axes[1].yaxis.set_major_formatter(plt.NullFormatter())

    return ax, axes


def ax_transits(ax, t, y, yerr=None, period=None, t0=None, duration=None, depth=None, transit_times={}, offset=0.2, plot_bin=False, 
                data_errorbar_args={'ms':2, 'fmt':'o', 'mew':1, 'alpha':1, 'zorder':2,},
                bin_data_args={'n_points': 200},
                bin_errorbar_args={},
                **kwargs):
    if len(transit_times) == 0:
        if (period is not None) and (t0 is not None):
            transit_times = get_transits_in_data(t, period, t0)
        else:
            raise Exception('Period/t0 or transit times must be provided')

    colour1 = Colour((122/255, 231/255, 199/255), 'sRGB')
    colour2 = Colour((224/255, 119/255, 125/255), 'sRGB')
    colour3 = Colour((244/255, 211/255, 94/255), 'sRGB')
    colour4 = Colour((172/255, 148/255, 247/255), 'sRGB')
    #print(colour1, colour2, colour3, colour4)
    gradient = Gradient([colour2, colour3, colour1, colour4], [0, .3, .7, 1], interp_space='Oklab')
    #print(gradient)

    if period is None:
        period = np.min(np.array(transit_times.values()).diff())
    if t0 is None:
        t0 = np.min(np.array(transit_times.values()))

    for i, n in enumerate(transit_times.keys()):
        min_orbit = transit_times[n] - period/2
        max_orbit = transit_times[n] + period/2
        lc_slice = np.logical_and(t >= min_orbit, t < max_orbit)

        position = (n - min(transit_times.keys())) / max(transit_times.keys())
        c = gradient.sample(position, position, 1, output_space='Hex', return_alpha=False)

        ax_lightcurve(ax, *fold_data(t[lc_slice], y[lc_slice]+(i*offset), yerr[lc_slice], period, t0),
                      c=c, mfc='w', mec=c,
                      data_errorbar_args=data_errorbar_args)

    if (period is not None) and (duration is not None):
        xlims = -duration/(period), duration/(period)
        ax.set_xlim(xlims)
    if depth is not None:
        ylims = -3*depth + 1, 2*depth + 1 + (len(transit_times)*offset)
        ax.set_ylim(ylims)


# ---------------------------------------------------------------------------- #
# Histograms                                                                   #
# ---------------------------------------------------------------------------- #

def ax_stacked_histogram(ax, x, y, xlims=None, ylims=None, xbins=40, ybins=40, cmap='viridis', label='', cbar_label=''):
    if xlims is None:
        xlims = (np.min(x), np.max(x))
    if ylims is None:
        ylims = (np.min(y), np.max(y))

    bottom = np.zeros(xbins)

    normal = mcolors.Normalize(*ylims)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normal)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(cbar_label, rotation=90)

    # Histogram (y) ---------------------------------------------------------- #
    y_counts, y_bins = np.histogram(y, bins=ybins, range=ylims)
    y_centre = (y_bins[:-1] + y_bins[1:]) / 2

    for i in range(ybins):
        if i == 0:
            xy_bins = x[y <= y_bins[i+1]]
        elif i == ybins - 1:
            xy_bins = x[y > y_bins[i]]
        else:
            xy_bins = x[y > y_bins[i]][y <= y_bins[i+1]]

        # Histogram (x) ---------------------------------------------------------- #
        x_counts, x_bins = np.histogram(xy_bins, bins=xbins, range=xlims)
        width = x_bins[1:] - x_bins[:-1]
        x_centre = (x_bins[:-1] + x_bins[1:]) / 2

        ax.bar(x_centre, x_counts, width=width*0.8, align='center', bottom=bottom,
            color=cmap(normal(np.abs(y_centre[i]))), label=label)
        
        bottom += x_counts
    
    ax.bar(x_centre, bottom, width=width*0.8, align='center',
           color='none', edgecolor='k', linewidth=0.5)
    
    ax.set_xlim(xlims)

# ---------------------------------------------------------------------------- #
# Periodograms                                                                 #
# ---------------------------------------------------------------------------- #

def ax_spectrum(ax, periods, power, period, planet=None, chunks=[]):
    vline_colour = '#40a1a1'

    if planet is not None:
        if (period >= planet.period.value - 0.01) and (period <= planet.period.value + 0.01):
            vline_colour = '#40a140'
        else:
            planet_colour = '#f04f4f'
            ax.axvline(planet.period.value, alpha=0.4, lw=3, c=planet_colour)
    
    ax.axvline(period, alpha=0.4, lw=3, c=vline_colour)
    
    for n in range(2, 10):
        ax.axvline(n*period, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)
        ax.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)

    for chunk in chunks:
        ax.axvline(chunk[0], linestyle=':', alpha=0.4, lw=1, c="#a1409f")

    ax.plot(periods, power, color=plt.rcParams['axes.edgecolor'], lw=0.5)

    ax.set_ylabel(r'SDE')
    ax.set_xlabel('Period (days)')
    ax.set_xlim(np.min(periods), np.max(periods))

def plot_spectrum(periods, power, period, planet=None, chunks=[], figsize=(10, 6), save=False, save_path=os.path.join(get_directory(), 'spectrum.png'), **kwargs):
    plot_axes(ax_spectrum, periods, power, period, planet, chunks, figsize=figsize, save=save, save_path=save_path, **kwargs)

# TLS ------------------------------------------------------------------------ #
def ax_TLS_spectrum(ax, tls_results, planet=None, chunks=[]):
    ax_spectrum(ax, tls_results.periods, tls_results.power, tls_results.period, planet, chunks)
    ax.set_title('TransitLeastSquares Power Spectrum')

def plot_TLS_spectrum(tls_results, planet=None, chunks=[], figsize=(10, 6), save=False, save_path=os.path.join(get_directory(), 'TLS_spectrum.png'), **kwargs):
    plot_axes(ax_TLS_spectrum, tls_results, planet, chunks, figsize=figsize, save=save, save_path=save_path, **kwargs)

# BLS ------------------------------------------------------------------------ #
def ax_BLS_spectrum(ax, bls_results, planet=None, chunks=[]):
    result_period = bls_results.period[np.argmax(bls_results.power)].value
    ax_spectrum(ax, bls_results.period, bls_results.power, result_period, planet, chunks)
    ax.set_title('BoxLeastSquares Power Spectrum')

def plot_BLS_spectrum(bls_results, planet=None, chunks=[], figsize=(10, 6), save=False, save_path=os.path.join(get_directory(), 'BLS_spectrum.png'), **kwargs):
    plot_axes(ax_BLS_spectrum, bls_results, planet, chunks, figsize=figsize, save=save, save_path=save_path, **kwargs)

# ---------------------------------------------------------------------------- #
# Dashboards                                                                   #
# ---------------------------------------------------------------------------- #

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

def title_text(ax, s, position=(0.5, 0.95), size='large', c='k', ec='w', **kwargs):
    ax.text(ax.get_xlim()[0] + (position[0] * (ax.get_xlim()[1] - ax.get_xlim()[0])),
            ax.get_ylim()[0] + (position[1] * (ax.get_ylim()[1] - ax.get_ylim()[0])),
            s, color=c, size=size, verticalalignment='top', horizontalalignment='center',
            path_effects=[withStroke(linewidth=3, foreground=ec, alpha=0.8)])

def TLS_dashboard(tls_results, star, lc, chunks=[], save=False, save_path='', **kwargs):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches

    fig = plt.figure(figsize=(1320*px, 800*px))#, layout='constrained')

    face_colours = 'none'
    edge_colours = plt.rcParams['axes.edgecolor']

    # Define gridspec and axes ----------------------------------------------- #
    gs = GridSpec(3, 5, fig, width_ratios=[1, 1, 1, 0.8, 0.8])

    ax_info         = fig.add_subplot(gs[:, 0])
    ax_spectrum     = fig.add_subplot(gs[0, 1:5])
    ax_phase        = fig.add_subplot(gs[1, 1:3])
    ax_odd_even     = fig.add_subplot(gs[2, 1:3])

    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1:3, 3:5], height_ratios=(1, 4))
    ax_neighborhood = fig.add_subplot(gs_right[1])
    ax_results = fig.add_subplot(gs_right[0])

    # Sort out other quantities ---------------------------------------------- #
    intransit_folded_model = np.where(tls_results['model_folded_model'] < 1.)[0]
    if len(intransit_folded_model) > 1:
        duration = tls_results['period'] * (tls_results['model_folded_phase'][intransit_folded_model[-1]]
                                         -  tls_results['model_folded_phase'][intransit_folded_model[0]])
    else:
        duration = tls_results['duration']

    snr = tls_results.snr / np.sqrt(tls_results.duration / duration)
    if snr >= 5:
        snr_colour = '#40a140'
    else:
        snr_colour = '#f04f4f'

    # Info box --------------------------------------------------------------- #
    ax_info.set_axis_off()
    ax_info.set_aspect(900/240)
    aspect_info = ax_info.get_aspect()

    # Make box patches
    inf_box = FancyBboxPatch((0, 0), 1, 1,
            boxstyle="round,pad=-0.0040,rounding_size=0.1",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='-',
            alpha=1, mutation_aspect=1/aspect_info
        )
    
    loc_box = FancyBboxPatch((10/240, 630/900), 220/240, 100/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='--',
            alpha=1, mutation_aspect=1/aspect_info
            )
    
    mag_box = FancyBboxPatch((10/240, 490/900), 220/240, 130/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='--',
            alpha=1, mutation_aspect=1/aspect_info
        )
    
    sta_box = FancyBboxPatch((10/240, 380/900), 220/240, 100/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='--',
            alpha=1, mutation_aspect=1/aspect_info
        )
    
    SDE_box = FancyBboxPatch((10/240, 240/900), 220/240, 130/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='-',
            alpha=1, mutation_aspect=1/aspect_info
        )
    
    res_box = FancyBboxPatch((10/240, 70/900), 220/240, 160/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=face_colours, linewidth=1, linestyle='-',
            alpha=1, mutation_aspect=1/aspect_info
        )
    
    snr_box = FancyBboxPatch((10/240, 10/900), 220/240, 50/900,
            boxstyle="round,pad=-0.0040,rounding_size=0.05",
            edgecolor=edge_colours, fc=snr_colour, linewidth=1, linestyle='-',
            alpha=1, mutation_aspect=1/aspect_info
        )

    # Add boxes
    ax_info.add_patch(inf_box)
    ax_info.add_patch(loc_box)
    ax_info.add_patch(mag_box)
    ax_info.add_patch(sta_box)
    ax_info.add_patch(SDE_box)
    ax_info.add_patch(res_box)
    ax_info.add_patch(snr_box)

    # Add text to infobox
    ax_info.text(0.5, 890/900,
                 star.spc, c='k', size='xx-large', verticalalignment='top', horizontalalignment='center',
                 path_effects=[withStroke(linewidth=3, foreground='w', alpha=0.8)])
    
    ax_info.text(0.5, 850/900,
                 f'TIC {star.name}', size='x-large', verticalalignment='top', horizontalalignment='center')
    
    ax_info.text(0.5, 820/900,
                 f'Gaia DR2:\n{star.gaia}', size='medium', verticalalignment='top', horizontalalignment='center')
    
    ax_info.text(0.5, 775/900,
                 f'2MASS:\n{star._2MASS}', size='medium', verticalalignment='top', horizontalalignment='center')

    ax_info.text(20/240, 640/900, f'RA:\nDec:\nDist:',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(220/240, 640/900, f'{star.ra:.7f}\n{star.dec:.7f}\n{star.distance.value:.0f}$\pm${star.distance.lower} pc',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)
    
    ax_info.text(0.5, 610/900,
                 f'Magnitudes', size='large', verticalalignment='top', horizontalalignment='center')

    ax_info.text(20/240, 500/900, f'TESS:\nI:\nH:',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(120/240, 500/900, f'{star.TESSmag:.1f}\n{star.Imag:.1f}\n{star.Hmag:.1f}',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)
    
    ax_info.text(140/240, 500/900, f'G:\nJ:\nK:',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(220/240, 500/900, f'{star.Gmag:.1f}\n{star.Jmag:.1f}\n{star.Kmag:.1f}',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)

    ax_info.text(20/240, 390/900, f'Mass ($\\text{{M}}_{{\odot}}$):\nRadius ($\\text{{R}}_{{\odot}}$):\nTeff (K):',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(220/240, 390/900, f'{star.mass.value:.2f}$\pm${star.mass.lower:.2f}\n{star.radius.value:.2f}$\pm${star.radius.lower:.2f}\n{star.temperature.value:.0f}$\pm${star.temperature.lower:.0f}',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)
    
    ax_info.text(20/240, 250/900, 'SDE:\nProminence:\nSDE raw:\nProm. raw:',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(220/240, 250/900, f'{tls_results.SDE:.2f}\n{tls_results.prominence:.2f}\n{tls_results.SDE_raw:.2f}\n{tls_results.prominence_raw:.2f}',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)
    
    ax_info.text(20/240, 80/900, 'Period (d):\nt0:\nDuration (h):\nDepth (%):\nRp/R*',
                size='medium', verticalalignment='bottom', horizontalalignment='left', linespacing=2)
    
    ax_info.text(220/240, 80/900, f'{tls_results.period:.8f}\n{tls_results.T0:.8f}\n{duration*24:.2f}\n{(1-tls_results.depth)*1e2:.2f}\n{tls_results.rp_rs:.4f}',
                size='medium', verticalalignment='bottom', horizontalalignment='right', linespacing=2)
    
    ax_info.text(120/240, 32.5/900, f'SNR = {snr:.2f}', color='w',
                size='x-large', verticalalignment='center', horizontalalignment='center', linespacing=1)

    # Periodogram ------------------------------------------------------------ #
    ax_TLS_spectrum(ax_spectrum, tls_results, chunks=chunks)

    ax_spectrum.set_title('')
    title_text(ax_spectrum, 'TransitLeastSquares Periodogram')

    # Nearby stars ----------------------------------------------------------- #
    from .tpfplotter.tpfplotter import plot_tpf

    #tic=system.name.replace('TIC ', '')
    plot_tpf(tic=str(star.name), ax=ax_neighborhood)
    # Phasefold -------------------------------------------------------------- #
    ax_phasefold(ax_phase, lc.t, lc.y, lc.e, period=tls_results.period, t0=tls_results.T0, bin_data_args={'t_bins': 0.001})
    residual_line(ax_phase, tls_results.depth_mean[0], tls_results.depth_mean[1])
    ax_phase.plot(tls_results.model_folded_phase - 0.5, tls_results.model_folded_model, c='#40A1A1', linestyle='-', zorder=10, path_effects=[Stroke(linewidth=3, foreground='w', alpha=0.5), Normal()])

    xlims = -duration/(tls_results.period), duration/(tls_results.period)
    ylims = -3*(tls_results.rp_rs)**2 + 1, 2*(tls_results.rp_rs)**2 + 1
    ax_phase.set_xlim(xlims)
    ax_phase.set_ylim(ylims)

    title_text(ax_phase, 'Phasefolded Signal')
    title_text(ax_phase, f'$\delta = {(1 - tls_results.depth_mean[0])*1e2:.1f} \pm {tls_results.depth_mean[1]*1e2:.1f}$ %', size='medium', position=(0.5, 0.85))

    # Results box ------------------------------------------------------------ #
    ax_lightcurve(ax_results, lc.t, lc.y, lc.e, transit_times=get_transits_in_data(lc.t, tls_results.period, tls_results.T0))

    ax_results.set_ylim([0.8, 1.2])

    # Odd-even transits ------------------------------------------------------ #
    ax_odd_even, axes = ax_oddeven(ax_odd_even, lc.t, lc.y, lc.e, period=tls_results.period, t0=tls_results.T0, bin_data_args={'t_bins': 0.002})

    residual_line(axes[0], tls_results.depth_mean_odd[0], tls_results.depth_mean_odd[1])
    residual_line(axes[1], tls_results.depth_mean_even[0], tls_results.depth_mean_even[1])

    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)

    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)

    title_text(axes[0], 'Odd')
    title_text(axes[1], 'Even')

    title_text(axes[0], f'$\delta = {(1 - tls_results.depth_mean_odd[0])*1e2:.1f} \pm {tls_results.depth_mean_odd[1]*1e2:.1f}$ %', size='medium', position=(0.5, 0.85))
    title_text(axes[1], f'$\delta = {(1 - tls_results.depth_mean_even[0])*1e2:.1f} \pm {tls_results.depth_mean_even[1]*1e2:.1f}$ %', size='medium', position=(0.5, 0.85))

    title_text(ax_odd_even, f'Odd-Even Mismatch: {tls_results.odd_even_mismatch:.2f} $\sigma$', position=(0.5, 1.1), size='large')

    #format_axes(fig)

    # Finish up -------------------------------------------------------------- #
    fig.tight_layout()

    if not save:
        plt.show()

    if save:
        savefig_args = extract_kwargs(plt.savefig, kwargs)
        plt.savefig(save_path, **savefig_args)
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .utils import extract_kwargs
from .data import bin_data
from .io import get_directory, find_path

def use_custom_styles():
    plt.style.use(find_path('styles/light_style.mplstyle'))
    plt.style.use(find_path('styles/use_serif.mplstyle'))
    plt.style.use(find_path('styles/paper_text.mplstyle'))

################################################################################
# - PLOTTING DATA ------------------------------------------------------------ #
################################################################################

# ---------------------------------------------------------------------------- #
# Lightcurve plotting                                                          #
# ---------------------------------------------------------------------------- #

def ax_lightcurve(ax, t, y, yerr=None, transit_times=[], plot_bin=True, data_errorbar_args={}, bin_errorbar_args={}, **kwargs):
    if ax == None:
        ax = plt.axes()

    if yerr == None:
        yerr = np.zeros(np.shape(y))
    
    ax.errorbar(t, y, yerr=yerr, ms=1, ls='none', c='#f04f4f', fmt='o', mfc='#f04f4f', mec='#4f2020', alpha=0.5, zorder=2, **data_errorbar_args)

    for n in transit_times:
        ax.axvline(transit_times[n], c='#40a1a1', alpha=0.5, zorder=0)

    if plot_bin:
        bin_data_args = extract_kwargs(bin_data, kwargs)
        t_bin, y_bin, yerr_bin = bin_data(t, y, yerr, **bin_data_args)

        ax.errorbar(t_bin, y_bin, yerr_bin, ms=4, capsize=2, elinewidth=1, fmt='o', mfc='w', mec='k', ecolor='k', zorder=20, **bin_errorbar_args)

    ax.set_xlabel('Time (BJD)')
    ax.set_ylabel('Flux')

    return ax

def plot_lightcurve(t, y, yerr=None, transit_times=[], plot_bin=True, save=False, save_path=None, **kwargs):
    fig = plt.figure()
    ax = fig.subplots()

    ax = ax_lightcurve(ax, t, y, yerr, transit_times, plot_bin, **kwargs)

    if not save:
        plt.show()

    if save:
        savefig_args = extract_kwargs(plt.savefig, kwargs)

        if save_path == None:
            dirname = get_directory()
            save_path = os.path.join(dirname, 'lightcurve.png')
        plt.savefig(save_path, **savefig_args)

# ---------------------------------------------------------------------------- #
# Phasefold plotting                                                           #
# ---------------------------------------------------------------------------- #



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

    print(sum(bottom))
    
    return ax
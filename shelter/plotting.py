import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import extract_kwargs
from .data import bin_data
from .io import get_directory

def use_custom_styles():
    plt.style.use(find_path('styles/light_style.mplstyle'))
    plt.style.use(find_path('styles/use_serif.mplstyle'))
    plt.style.use(find_path('styles/paper_text.mplstyle'))

###########################################################################
# - PLOTTING DATA ------------------------------------------------------- #
###########################################################################

# ----------------------------------------------------------------------- #
#  Lightcurve plotting                                                    #
# ----------------------------------------------------------------------- #

def ax_lightcurve(ax, t, y, yerr=None, transit_times=[], plot_bin=True, **kwargs):
    if ax == None:
        ax = plt.axes()

    if yerr == None:
        yerr = np.zeros(np.shape(y))

    ax.errorbar(t, y, yerr, c='#f04f4f', s=1, alpha=0.5, zorder=1)

    for n in transit_times:
        ax.axvline(transit_times[n], c='#40a1a1', alpha=0.5, zorder=0)

    if plot_bin:
        bin_data_args = extract_kwargs(bin_data, kwargs)
        t_bin, y_bin, yerr_bin = bin_data(t, y, yerr, **bin_data_args)

        ax.errorbar(t_bin, y_bin, yerr_bin, c='w', edgecolor='k', linewidths=1, s=10, zorder=2)

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

# ----------------------------------------------------------------------- #
#  Phasefold plotting                                                    #
# ----------------------------------------------------------------------- #
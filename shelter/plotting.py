import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .utils import extract_kwargs
from .data import bin_data, fold_data
from .io import get_directory, find_path

def use_custom_styles():
    plt.style.use(find_path('styles/light_style.mplstyle'))
    plt.style.use(find_path('styles/use_serif.mplstyle'))
    plt.style.use(find_path('styles/paper_text.mplstyle'))

################################################################################
# - PLOTTING DATA ------------------------------------------------------------ #
################################################################################

def plot_axes(func, *args, save=False, save_path='', **kwargs):
    fig = plt.figure()
    ax = fig.subplots()

    ax = func(ax, *args)

    if not save:
        plt.show()

    if save:
        savefig_args = extract_kwargs(plt.savefig, kwargs)
        plt.savefig(save_path, **savefig_args)

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
    if save_path == None:
        dirname = get_directory()
        save_path = os.path.join(dirname, 'lightcurve.png')
        
    plot_axes(ax_lightcurve, t, y, yerr, transit_times, plot_bin, save=save, save_path=save_path, **kwargs)

# ---------------------------------------------------------------------------- #
# Phasefold plotting                                                           #
# ---------------------------------------------------------------------------- #

def ax_phasefold(ax, t, y, yerr=None, period=None, t0=None, ):
    return

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

# ---------------------------------------------------------------------------- #
# Periodograms                                                                 #
# ---------------------------------------------------------------------------- #

def ax_TLS_spectrum(ax, tls_results, planet=None):
    vline_colour = '#40a1a1'

    if planet is not None:
        if (tls_results.period >= planet.period.value - 0.01) and (tls_results.period <= planet.period.value + 0.01):
            vline_colour = '#40a140'
        else:
            planet_colour = '#f04f4f'
            ax.axvline(planet.period.value, alpha=0.4, lw=3, c=planet_colour)
    
    ax.axvline(tls_results.period, alpha=0.4, lw=3, c=vline_colour)
    
    for n in range(2, 10):
        ax.axvline(n*tls_results.period, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)
        ax.axvline(tls_results.period / n, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)
        
    ax.plot(tls_results.periods, tls_results.power, color='black', lw=0.5)

    ax.set_ylabel(r'SDE')
    ax.set_xlabel('Period (days)')
    ax.set_xlim(np.min(tls_results.periods), np.max(tls_results.periods))
    ax.set_title('TransitLeastSquares Power Spectrum')

    return ax

def plot_TLS_spectrum(tls_results, planet=None, save=False, save_path=None, **kwargs):
    if save_path == None:
        dirname = get_directory()
        save_path = os.path.join(dirname, 'TLS_spectrum.png')

    plot_axes(ax_TLS_spectrum, tls_results, planet, save=save, save_path=save_path, **kwargs)

def ax_BLS_spectrum(ax, bls_results, planet=None):
    vline_colour = '#40a1a1'
    result_period = bls_results.period[np.argmax(bls_results.power)].value
    
    if planet is not None:
        if (result_period >= planet.period.value - 0.01) and (result_period <= planet.period.value + 0.01):
            vline_colour = '#40a140'
        else:
            planet_colour = '#f04f4f'
            ax.axvline(planet.period.value, alpha=0.4, lw=3, c=planet_colour)
    
    ax.axvline(result_period, alpha=0.4, lw=3, c=vline_colour)
    
    for n in range(2, 10):
        ax.axvline(n*result_period, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)
        ax.axvline(result_period / n, alpha=0.4, lw=1, linestyle="dashed", c=vline_colour)

    ax.plot(bls_results.period.value, bls_results.power, color='black', lw=0.5)

    ax.set_ylabel(r'SDE')
    ax.set_xlabel('Period (days)')
    ax.set_xlim(np.min(bls_results.period.value), np.max(bls_results.period.value))
    ax.set_title('BoxLeastSquares Power Spectrum')

    return ax

def plot_BLS_spectrum(bls_results, planet=None, save=False, save_path=None, **kwargs):
    if save_path == None:
        dirname = get_directory()
        save_path = os.path.join(dirname, 'BLS_spectrum.png')

    plot_axes(ax_BLS_spectrum, bls_results, planet, save=save, save_path=save_path, **kwargs)
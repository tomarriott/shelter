"""
tpfplotter.py

Callable (importable) version of the tpfplotter CLI script.

Instead of parsing sys.argv via argparse, the two functions below take
regular Python arguments so you can call them from another script, e.g.:

    from tpfplotter import plot_tpf

    plot_tpf(tic="123456789", maglim=6)

or, for a list of targets:

    from tpfplotter import run_tpfplotter

    run_tpfplotter(list_file="targets.txt")x
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.visualization as stretching
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.table import Table
from astropy.io import ascii

# --- Update this import to point at wherever these helpers live ---
from .tpfplotter_og import (
    get_coord,
    get_gaia_data,
    get_dr2_id_from_tic,
    dr3_from_dr2,
    add_gaia_figure_elements,
    plot_orientation,
    search_tesscut,
    search_targetpixelfile,
)

font_large = 12
font_medium = 10
font_small = 10

def plot_tpf(
    tic=None,
    coord=None,
    sector=None,
    name=None,
    maglim=6.0,
    gid=None,
    gmag=None,
    legend="best",
    pm=False,
    savegaia=False,
    sradius=60,
    foldname="./",
    ax=None,
    save=None,
):
    """
    Produce a single TPF + Gaia-sources plot for one target.

    Parameters
    ----------
    tic : str or None
        TIC number (as a string), e.g. "123456789". Required unless `coord`
        is given.
    coord : str or None
        "ra,dec" string (comma separated) to search by coordinates instead
        of TIC number, e.g. "10.6847,41.2687". If given, this takes
        precedence over `tic` for the search, but `tic` is still used for
        labeling output filenames if provided.
    sector : int or None
        TESS sector to use. If None, the first available sector is used.
    name : str or None
        Custom display name for the target (used in plot title / filenames).
    maglim : float
        Magnitude range (relative to target) of Gaia sources to plot.
    gid : str or int or None
        Gaia DR3 source ID, if already known (skips the TIC/coord -> Gaia
        lookup).
    gmag : float or None
        Gaia magnitude of the target, required if `gid` is given.
    legend : str
        Matplotlib legend location string, e.g. "best", "upper right".
    pm : bool
        If True, draw proper-motion arrows for nearby Gaia sources.
    savegaia : bool
        If True, also save a table of nearby Gaia sources to a .dat file.
    sradius : float
        Search radius (arcsec) used when looking up Gaia sources by
        coordinates.
    foldname : str
        Output directory to save the plot (and Gaia table) into.
    ax : matplotlib.axes.Axes or None
        If given, draw the TPF image, aperture, Gaia sources, labels, etc.
        onto this existing Axes instead of creating a new figure. Useful for
        putting the TPF plot into a subplot grid alongside other panels.
        The colorbar is attached to `ax` via an appended axes (so it will
        sit right next to whichever subplot `ax` is), and the figure is
        taken from `ax.figure`.
    save : bool or None
        Whether to call `plt.savefig(...)` (and `plt.close(fig)`) at the
        end. Defaults to `True` when `ax` is None (original behaviour: this
        function owns the figure) and `False` when an external `ax` is
        passed in (the caller owns the figure and decides when/if to save
        it). Pass an explicit True/False to override this default.

    Returns
    -------
    str or None
        Path to the saved PDF plot, or None if `save` was False.
    """
    if tic is None and coord is None:
        raise ValueError("You must provide either `tic` or `coord`.")

    use_coord = coord is not None
    tic_label = str(tic) if tic is not None else "COORD"

    if use_coord:
        ra, dec = coord.split(",")[0], coord.split(",")[1]
        print("* Making TPFplot for " + tic_label + " (ra = " + ra + ", " + "dec = " + dec + ") ...")
    else:
        ra, dec = get_coord(tic_label)
        print("* Making TPFplot for TIC" + tic_label + " (ra = " + str(ra) + ", " + "dec = " + str(dec) + ") ...")

    # --- Resolve Gaia ID / magnitude -------------------------------------
    if gid is not None:
        gaia_id, mag = gid, float(gmag)
    else:
        if use_coord:
            gaia_id, mag = get_gaia_data(ra, dec, search_radius=sradius, DR2=False)
        else:
            gaia_id, mag = get_dr2_id_from_tic(tic_label)
            gaia_id = dr3_from_dr2(gaia_id)
            if np.isnan(mag):
                gaia_id, mag = get_gaia_data(ra, dec, search_radius=sradius, DR2=False)

    # --- Download the TPF --------------------------------------------------
    if use_coord:
        if sector is not None:
            tpf = search_tesscut(ra + " " + dec, sector=int(sector)).download(cutout_size=(12, 12))
        else:
            tpf = search_tesscut(ra + " " + dec).download(cutout_size=(12, 12))
        pipeline = "False"
        print("\t --> Using TESScut to get the TPF")
    else:
        try:
            if sector is not None:
                tpf = search_targetpixelfile("TIC " + tic_label, sector=int(sector), mission="TESS").download()
            else:
                tpf = search_targetpixelfile("TIC " + tic_label, mission="TESS").download()
            _ = tpf.flux  # sanity check it has a flux array
            pipeline = "True"
            print("\t --> Target found in the CTL!")
        except Exception:
            if sector is not None:
                tpf = search_tesscut("TIC " + tic_label, sector=int(sector)).download(cutout_size=(12, 12))
            else:
                tpf = search_tesscut("TIC " + tic_label).download(cutout_size=(12, 12))
            print("\t --> Target not in CTL. The FFI cut out was successfully downloaded")
            pipeline = "False"

    # --- Set up figure / axes -------------------------------------------
    owns_figure = ax is None
    if save is None:
        save = owns_figure

    if owns_figure:
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(1, 3, height_ratios=[1], width_ratios=[1, 0.05, 0.01])
        gs.update(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.03)
        ax1 = plt.subplot(gs[0, 0])
    else:
        ax1 = ax
        fig = ax1.figure

    plt.sca(ax1)

    # TPF plot
    mean_tpf = np.mean(tpf.flux, axis=0)
    nx, ny = np.shape(mean_tpf)
    norm = ImageNormalize(stretch=stretching.LogStretch())
    division = int(np.log10(np.nanmax(np.nanmean(tpf.flux.value, axis=0))))
    image = np.nanmean(tpf.flux, axis=0) / 10 ** division
    splot = plt.imshow(
        image.value,
        norm=norm,
        extent=[tpf.column - 0.5, tpf.column + ny - 0.5, tpf.row - 0.5, tpf.row + nx - 0.5],
        origin="lower",
        zorder=0,
    )

    # Pipeline aperture
    if pipeline == "True":
        aperture_mask = tpf.pipeline_mask
        aperture = tpf._parse_aperture_mask(aperture_mask)
        maskcolor = "tomato"
        print("\t --> Using pipeline aperture...")
    else:
        aperture_mask = tpf.create_threshold_mask(threshold=3, reference_pixel=(int(nx / 2), int(ny / 2)))
        aperture = tpf._parse_aperture_mask(aperture_mask)
        maskcolor = "lightgray"
        print("\t --> Using threshold aperture...")

    for i in range(aperture.shape[0]):
        for j in range(aperture.shape[1]):
            if aperture_mask[i, j]:
                ax1.add_patch(
                    patches.Rectangle(
                        (j + tpf.column - 0.5, i + tpf.row - 0.5), 1, 1, color=maskcolor, fill=True, alpha=0.4
                    )
                )
                ax1.add_patch(
                    patches.Rectangle(
                        (j + tpf.column - 0.5, i + tpf.row - 0.5), 1, 1, color=maskcolor, fill=False, alpha=1, lw=2
                    )
                )

    # Gaia sources
    r, res = add_gaia_figure_elements(tpf, magnitude_limit=mag + float(maglim), targ_mag=mag, gaia_id=gaia_id, DR2=False)
    x, y, gaiamags, xpm, ypm = r
    x, y, gaiamags, xarrow, yarrow = np.array(x), np.array(y), np.array(gaiamags), np.array(xpm), np.array(ypm)
    size = 128.0 / 2 ** ((gaiamags - mag))
    plt.scatter(x, y, s=size, c="red", alpha=0.6, edgecolor='w', zorder=10)
    if pm:
        for i in range(len(x)):
            plt.arrow(
                x[i], y[i], xpm[i] - x[i], (ypm[i] - y[i]),
                head_width=0.1, head_length=0.15, overhang=0.2, color="gray", alpha=0.8,
            )

    # Gaia source for the target
    this = np.where(np.array(res["Source"]) == int(gaia_id))[0]
    plt.scatter(x[this], y[this], marker="x", c="white", s=32, zorder=11)

    # Legend
    add = 0
    if int(maglim) % 2 != 0:
        add = 1
    maxmag = int(maglim) + add
    legend_mags = np.linspace(-2, maxmag, int((maxmag + 2) / 2 + 1))
    fake_sizes = mag + legend_mags
    for f in fake_sizes:
        size = 128.0 / 2 ** ((f - mag))
        plt.scatter(0, 0, s=size, c="red", alpha=0.6, edgecolor='w', zorder=10, label=r"$\Delta m=$ " + str(int(f - mag)))

    ax1.legend(fancybox=True, framealpha=0.7, loc=legend, fontsize=font_small, labelcolor='w')

    # Source labels
    dist = np.sqrt((x - x[this]) ** 2 + (y - y[this]) ** 2)
    dsort = np.argsort(dist)
    corners = np.array(
        [
            np.abs(x[this] - (tpf.column + nx)),
            np.abs(x[this] - tpf.column),
            np.abs(y[this] - (tpf.row + ny)),
            np.abs(y[this] - tpf.row),
        ]
    )
    xmin = tpf.column + 0.05 * nx
    xmax = tpf.column + 0.95 * nx
    ymin = tpf.row + 0.05 * ny
    ymax = tpf.row + 0.95 * ny
    for d, elem in enumerate(dsort):
        if (x[elem] < xmax) & (x[elem] > xmin) & (y[elem] < ymax) & (y[elem] > ymin):
            plt.text(x[elem] + 0.1, y[elem] + 0.1, str(d + 1), color="white", zorder=100, fontsize=font_small)

    # Orientation arrows
    plot_orientation(tpf)

    # Labels and titles (reverse x limits so the image plots as seen on sky)
    plt.xlim(tpf.column + ny - 0.5, tpf.column - 0.5)
    plt.ylim(tpf.row - 0.5, tpf.row + nx - 0.5)
    plt.xlabel("Pixel Column Number", fontsize=font_medium, zorder=200)
    plt.ylabel("Pixel Row Number", fontsize=font_medium, zorder=200)
    if use_coord:
        plt.title("Coordinates " + tic_label + " - Sector " + str(tpf.sector), fontsize=font_large, zorder=200)
    elif name:
        plt.title(str(name) + " - Sector " + str(tpf.sector), fontsize=font_large, zorder=200)
    else:
        plt.title("TIC " + tic_label + " - Sector " + str(tpf.sector), fontsize=font_large, zorder=200)

    # Colorbar (appended to ax1 so this works whether ax1 came from our own
    # gridspec or was passed in externally)
    divider = make_axes_locatable(ax1)
    cbax = divider.append_axes("right", size="5%", pad=0.1)

    cb = Colorbar(ax=cbax, mappable=splot, orientation="vertical", ticklocation="right")
    plt.xticks(fontsize=font_medium)
    exponent = r"$\times 10^" + str(division) + "$"
    cb.set_label(r"Flux " + exponent + r" (e$^-$/s)", labelpad=10, fontsize=font_medium)

    othername = ("_" + name.split("/")[0]) if name else ""
    outpath = None
    if save:
        os.makedirs(foldname, exist_ok=True)
        outpath = os.path.join(foldname, "TPF_Gaia" + othername + "_TIC" + tic_label + "_S" + str(tpf.sector) + ".pdf")
        fig.savefig(outpath)
        print("\t --> TPF plot written in file: " + outpath)
    if owns_figure:
        plt.close(fig)

    # Save Gaia sources info
    if savegaia:
        dist = np.sqrt((x - x[this]) ** 2 + (y - y[this]) ** 2)
        GaiaID = np.array(res["Source"])
        srt = np.argsort(dist)
        x_s, y_s, gaiamags_s, dist_s, GaiaID_s = x[srt], y[srt], gaiamags[srt], dist[srt], GaiaID[srt]

        IDs = np.arange(len(x_s)) + 1
        inside = np.zeros(len(x_s))
        for i in range(aperture.shape[0]):
            for j in range(aperture.shape[1]):
                if aperture_mask[i, j]:
                    xtpf, ytpf = j + tpf.column, i + tpf.row
                    _inside = np.where((x_s > xtpf) & (x_s < xtpf + 1) & (y_s > ytpf) & (y_s < ytpf + 1))[0]
                    inside[_inside] = 1

        data = Table(
            [IDs, GaiaID_s, x_s, y_s, dist_s, dist_s * 21.0, gaiamags_s, inside.astype("int")],
            names=["# ID", "GaiaID", "x", "y", "Dist_pix", "Dist_arcsec", "Gmag", "InAper"],
        )
        gaia_outpath = os.path.join(foldname, "Gaia" + othername + "_TIC" + tic_label + "_S" + str(tpf.sector) + ".dat")
        ascii.write(data, gaia_outpath, overwrite=True)
        print("\t --> Gaia close sources saved in file: " + gaia_outpath)

    print("\t --> Done!\n")
    return outpath


def run_tpfplotter(
    tic=None,
    coord=None,
    sector=None,
    name=None,
    maglim=6.0,
    gid=None,
    gmag=None,
    legend="best",
    pm=False,
    savegaia=False,
    sradius=60,
    list_file=None,
):
    """
    High-level entry point mirroring the original CLI's behaviour.

    Either pass a single target via `tic` (or `coord`), or pass a whitespace
    -delimited `list_file` with a header row containing at least a `tic`
    column (and optionally `ra`, `dec`, `sector`, `maglim`, `name` columns),
    exactly like the original `--list` CLI option.

    Returns
    -------
    list of str
        Paths to all saved plot PDFs.
    """
    outputs = []

    if list_file is not None:
        print("* Using file " + list_file + " as the list of requested targets *")
        tab = pd.read_table(list_file, delimiter=" ", header=0)
        tab_colnames = tab.columns.values
        tics = tab["tic"].values
        n_targets = len(tics)

        sectors = np.array([None for _ in range(n_targets)])
        names = np.array([None for _ in range(n_targets)])
        maglims = np.zeros(n_targets) + maglim
        coords_list = [None] * n_targets

        if "ra" in tab_colnames and "dec" in tab_colnames:
            print("\t\t --> RA and DEC columns found in file, using COORDS to search for target.")
            ras, decs = tab["ra"].values, tab["dec"].values
            coords_list = [f"{r},{d}" for r, d in zip(ras, decs)]
        if "sector" in tab_colnames:
            print("\t\t --> SECTOR column found in file, using requested sectors for each target")
            sectors = tab["sector"].values
        if "maglim" in tab_colnames:
            print("\t\t --> MAGLIM column found in file, using custom (requested) maglim for each target")
            maglims = tab["maglim"].values
        if "name" in tab_colnames:
            print("\t\t --> NAME column found in file, using custom (requested) name in plot for each target")
            names = tab["name"].values
        print("\n")

        foldname = os.path.splitext(list_file)[0]
        os.makedirs(foldname, exist_ok=True)

        for tt, this_tic in enumerate(tics):
            outpath = plot_tpf(
                tic=str(this_tic),
                coord=coords_list[tt],
                sector=sectors[tt],
                name=names[tt],
                maglim=maglims[tt],
                gid=gid,
                gmag=gmag,
                legend=legend,
                pm=pm,
                savegaia=savegaia,
                sradius=sradius,
                foldname=foldname,
            )
            outputs.append(outpath)

    else:
        if tic is None and coord is None:
            raise ValueError("You must provide either `tic`, `coord`, or `list_file`.")
        outpath = plot_tpf(
            tic=tic,
            coord=coord,
            sector=sector,
            name=name,
            maglim=maglim,
            gid=gid,
            gmag=gmag,
            legend=legend,
            pm=pm,
            savegaia=savegaia,
            sradius=sradius,
            foldname="./",
        )
        outputs.append(outpath)

    return outputs


if __name__ == "__main__":
    # Example usage when run directly:
    # run_tpfplotter(tic="123456789", maglim=6)
    pass
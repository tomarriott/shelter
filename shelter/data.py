import numpy as np
# from dataclasses import dataclass, field

###########################################################################
# - DATA CLASSES -------------------------------------------------------- #
###########################################################################

# Merge RV and LC classes
# TODO: Make these dataclasses?
class DataContainer:
    def __init__(self, t, y, e=[], instrument=None):

        if len(t) != len(y):
            raise ValueError('t and y must be same size')

        self.t = np.asanyarray(t)
        self.y = np.asanyarray(y)
        if len(e) != len(y):
            e = np.pad(e, (0, len(y) - len(e))).reshape(len(y))
        self.e = e

        self.N = len(t)
        self.instrument = instrument

# ----------------------------------------------------------------------- #
#  Lightcurve class                                                       #
# ----------------------------------------------------------------------- #

class LightCurve(DataContainer):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return f"LightCurve(N_points={self.N}, Instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"

    def to_lightkurve(self, **kwargs):
        return to_lightkurve(self, **kwargs)

# ----------------------------------------------------------------------- #
#  Radial velocity class                                                  #
# ----------------------------------------------------------------------- #

class RadialVelocity(DataContainer):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return f"RadialVelocity(N_points={self.N}, Instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"
    
def to_lightkurve(lc, **kwargs):
    try:
        from lightkurve import LightCurve as lk_LightCurve
    except ImportError:
        print('Lightkurve is not installed! Skipping construction')
        return lc

    return lk_LightCurve(time=lc.t, flux=lc.y, flux_err=lc.e, **kwargs)

def from_lightkurve(lc):
    return LightCurve(lc.time, lc.flux, lc.flux_err)

# TODO: this was written by Claude - rewrite properly
def bin_data(t, y, yerr=None, n_points=None, n_bins=None, t_bins=None, method="mean"):
    '''
    Bin a sequential dataset into averaged bins.

    Exactly one of 'n_points', 'n_bins', or 't_bins' must be provided.
    Within each bin the central value is computed by either
    a weighted mean or a median, and an uncertainty is returned.

    Parameters
    ----------
    t : array-like
        Time (or x-axis) values of the dataset. Must be 1-D and the same length
        as y and yerr. Need not be uniformly spaced, but should be sorted in
        ascending order (or at least monotonically increasing).
    y : array-like
        Data values corresponding to each point in t.
    yerr : array-like, optional
        Measurement uncertainties (1-sigma) for each point in y. Used for
        weighted-mean calculations and propagated into the output uncertainties.
        Must be strictly positive.
    n_points : int, optional
        Fixed number of input data points to place in each bin.  The last bin
        may contain fewer points if len(t) is not exactly divisible by
        n_points.
    n_bins : int, optional
        Total number of equally-spaced (in time) bins to divide the data into.
        Bin edges are placed uniformly between t[0] and t[-1].
    t_bins : float, optional
        Width of each bin in the same units as t. Bins are built starting from
        t[0] and advancing by t_bins until all data are covered.
    method : {'mean', 'median'}, optional
        Averaging method to use within each bin (default 'mean').

        * 'mean': inverse-variance weighted mean; uncertainty is the
          quadrature combination of individual errors divided by N (i.e. the
          standard error of the weighted mean).
        * 'median': sample median; uncertainty is estimated as
          1.4826 * MAD / sqrt(N) (asymptotically equivalent to the standard
          error of the mean for Gaussian noise).

    Returns
    -------
    t_out : np.ndarray
        Bin centres (weighted mean of t values inside the bin, or simple mean
        when method='median').
    y_out : np.ndarray
        Binned data values.
    yerr_out : np.ndarray
        Uncertainties on the binned data values.

    Raises
    ------
    ValueError
        If not exactly one binning scheme is specified, if an unrecognised
        method is given, or if the inputs have inconsistent shapes / values.
    '''
    # ------------------------------------------------------------------- #
    #  Input validation                                                   #
    # ------------------------------------------------------------------- #
    if yerr == None:
        yerr = np.zeros(np.shape(y))
        no_yerr = True
    else:
        no_yerr = False

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if t.ndim != 1 or y.ndim != 1 or yerr.ndim != 1:
        raise ValueError("t, y, and yerr must all be 1-D arrays.")
    if not (t.shape == y.shape == yerr.shape):
        raise ValueError("t, y, and yerr must have the same length.")
    if not no_yerr:
        if np.any(yerr <= 0):
            raise ValueError("All yerr values must be strictly positive.")

    n_specified = sum(x is not None for x in (n_points, n_bins, t_bins))
    if n_specified != 1:
        raise ValueError(
            "Exactly one of n_points, n_bins, or t_bins must be specified "
            f"(got {n_specified})."
        )

    method = method.lower()
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', got '{method}'.")

    N = len(t)

    # ------------------------------------------------------------------- #
    #  Build a list of index slices, one per bin                          #
    # ------------------------------------------------------------------- #
    slices = []  # each element is a boolean mask or array of indices

    if n_points is not None:
        # --- fixed number of data-points per bin ------------------------
        n_points = int(n_points)
        if n_points < 1:
            raise ValueError("n_points must be >= 1.")
        starts = range(0, N, n_points)
        slices = [np.arange(s, min(s + n_points, N)) for s in starts]

    elif n_bins is not None:
        # --- fixed total number of (time-equal) bins --------------------
        n_bins = int(n_bins)
        if n_bins < 1:
            raise ValueError("n_bins must be >= 1.")
        edges = np.linspace(t[0], t[-1], n_bins + 1)
        # right-open intervals [edge_i, edge_{i+1}) except the last which
        # is closed on the right so that t[-1] always falls in a bin.
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (t >= edges[i]) & (t < edges[i + 1])
            else:
                mask = (t >= edges[i]) & (t <= edges[i + 1])
            if mask.any():
                slices.append(np.where(mask)[0])

    else:
        # --- fixed time-span per bin ------------------------------------
        if t_bins <= 0:
            raise ValueError("t_bins must be > 0.")
        left = t[0]
        while left < t[-1] + t_bins:          # ensure the last point is caught
            right = left + t_bins
            mask = (t >= left) & (t < right)
            if mask.any():
                slices.append(np.where(mask)[0])
            elif left > t[-1]:                # no more data ahead
                break
            left = right

    # ------------------------------------------------------------------- #
    #  Compute bin statistics                                             #
    # ------------------------------------------------------------------- #
    t_out, y_out, yerr_out = [], [], []

    for idx in slices:
        t_bin   = t[idx]
        y_bin   = y[idx]
        err_bin = yerr[idx]
        n       = len(idx)

        if method == "mean":
            # Inverse-variance weighted mean
            if no_yerr:
                weights  = np.ones(len(t_bin))
            else:
                weights  = 1.0 / err_bin**2
            w_sum        = weights.sum()
            t_centre     = np.sum(weights * t_bin) / w_sum
            y_centre     = np.sum(weights * y_bin) / w_sum
            # Uncertainty: sqrt(1 / sum(w_i))  – standard error of weighted mean
            y_err_centre = np.sqrt(1.0 / w_sum)

        else:  # median
            t_centre     = t_bin.mean()          # simple mean of times
            y_centre     = np.median(y_bin)
            # Robust uncertainty: 1.4826 * MAD / sqrt(N)
            # Falls back to mean(yerr)/sqrt(N) when N == 1
            if n > 1:
                mad          = np.median(np.abs(y_bin - y_centre))
                y_err_centre = 1.4826 * mad / np.sqrt(n)
                # If MAD is exactly zero (e.g. all identical values),
                # fall back to propagated individual errors.
                if y_err_centre == 0.0:
                    y_err_centre = np.sqrt(np.sum(err_bin**2)) / n
            else:
                y_err_centre = err_bin[0]

        t_out.append(t_centre)
        y_out.append(y_centre)
        yerr_out.append(y_err_centre)

    if no_yerr:
        return np.array(t_out), np.array(y_out),
    else:
        return np.array(t_out), np.array(y_out), np.array(yerr_out)

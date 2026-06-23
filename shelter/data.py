import os.path
import numpy as np
from .utils import to_list_of_arrays, get_epoch
# from dataclasses import dataclass, field

################################################################################
# - DATA CLASSES ------------------------------------------------------------- #
################################################################################

# Merge RV and LC classes
# TODO: Make these dataclasses?
class TimeSeries:
    def __init__(self, t, y, e=[], instrument='none'):

        if len(t) != len(y):
            raise ValueError('t and y must be same size')

        self.t = np.asanyarray(t)
        self.y = np.asanyarray(y)
        if len(e) != len(y):
            e = np.pad(e, (np.std(y), len(y) - len(e))).reshape(len(y))
        self.e = e

        self.N = len(t)
        self.instrument = instrument

    def __len__(self):
        return len(self.t)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self.t[index], self.y[index], self.e[index]
        elif isinstance(id, slice):
            return type(self)(self.t[index], self.y[index], self.e[index], self.instrument)
        
        elif all([isinstance(i, (bool, np.bool_)) for i in index]):
            if len(index) != len(self.t):
                raise IndexError(
                    f"boolean index did not match indexed array; dimension is {len(self.t)} "
                    f"but corresponding boolean dimension is {len(index)}"
                )
            return type(self)([self.t[i] for i in np.nonzero(index)[0]], [self.y[i] for i in np.nonzero(index)[0]], [self.e[i] for i in np.nonzero(index)[0]], self.instrument)
        elif all([isinstance(i, (int, np.integer)) for i in index]):
            # case int array like, follow ndarray behavior
            return type(self)([self.t[i] for i in np.nonzero(index)[0]], [self.y[i] for i in np.nonzero(index)[0]], [self.e[i] for i in np.nonzero(index)[0]], self.instrument)
        else:
            raise IndexError(
                "only integers, slices (`:`) and integer or boolean arrays are valid indices"
            )

    def append(self, t, y, e=[]):
        self.t = np.append(self.t, t)
        self.y = np.append(self.y, y)
        if len(e) != len(y):
            e = np.pad(e, (np.std(y), len(y) - len(e))).reshape(len(y))
        self.e = np.append(self.e, e)

    def order_data(self):
        self.t, self.y, self.e = order_data(self.t, self.y, self.e)

    def fold(self, period, t0=None):
        fold_t, fold_y, fold_e = fold_data(self.t, self.y, self.e, period, t0)
        folded_lightcurve = LightCurve(fold_t, fold_y, fold_e, self.instrument)
        folded_lightcurve.period = period
        folded_lightcurve.t0 = t0

        return folded_lightcurve
    
    def bin(self, n_points=None, n_bins=None, t_bins=None, method='mean'):
        bin_t, bin_y, bin_e = bin_data(*self.order_data(), n_points, n_bins, t_bins, method)
        binned_lightcurve = LightCurve(bin_t, bin_y, bin_e, self.instrument)

        return binned_lightcurve
    
    def to_juliet(self):
        return ({self.instrument: self.t}, {self.instrument: self.y}, {self.instrument: self.e})

# ---------------------------------------------------------------------------- #
# Lightcurve class                                                             #
# ---------------------------------------------------------------------------- #

class LightCurve(TimeSeries):
    def __init__(self, t, y, e=[], instrument='none', cadence=None, sector=None):
        super().__init__(t, y, e, instrument)

        if cadence is None:
            cadence = int((t[1] - t[0]) * (3600 * 24))
        self.cadence = cadence
        self.sector = sector

    def __repr__(self):
        return f"LightCurve(N={self.N}, instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"

    def to_lightkurve(self, **kwargs):
        return to_lightkurve(self, **kwargs)
    
    def flatten(self, window, function='wotan', **kwargs):
        
        if isinstance(window, float) or isinstance(window, int):
            window = [window]
            single = True
        
        # object to store lightcurves in if multiple window lengths ---------- #
        return_obj = DataCollection()

        if function == 'wotan':
            try:
                import wotan
            except ImportError:
                print('Wotan is not installed! Using [method] for now.')

            for wind in window:
                y_flat = wotan.flatten(self.t, self.y, window_length=wind, **kwargs)
                return_obj.append(LightCurve(self.t, y_flat, self.e, self.instrument, self.cadence))

        if single:
            return return_obj[0]
        return return_obj
    
    def mask_transits(self, ):
        return

# ---------------------------------------------------------------------------- #
# Radial velocity class                                                        #
# ---------------------------------------------------------------------------- #

class RadialVelocity(TimeSeries):
    def __init__(self, t, y, e=[], instrument='none'):
        super().__init__(t, y, e, instrument)

    def __repr__(self):
        return f"RadialVelocity(N_points={self.N}, Instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"

# ---------------------------------------------------------------------------- #
# Collection class                                                             #
# ---------------------------------------------------------------------------- #

class DataCollection:
    def __init__(self, data=[]):
        if not isinstance(data, list):
            data = [data]

        self.data = []
        for datum in data:
            self.data.append(datum)

    def __repr__(self):
        return f"DataCollection({[datum for datum in self.data]})"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return self.data[index]
        elif isinstance(id, slice):
            return type(self)(self.data[index])
        
        elif all([isinstance(i, (bool, np.bool_)) for i in index]):
            if len(index) != len(self.data):
                raise IndexError(
                    f"boolean index did not match indexed array; dimension is {len(self.data)} "
                    f"but corresponding boolean dimension is {len(index)}"
                )
            return type(self)([self.data[i] for i in np.nonzero(index)[0]])
        elif all([isinstance(i, (int, np.integer)) for i in index]):
            # case int array like, follow ndarray behavior
            return type(self)([self.data[i] for i in index])
        else:
            raise IndexError(
                "only integers, slices (`:`) and integer or boolean arrays are valid indices"
            )
        
    def __setitem__(self, index, item):
        self.data[index] = item

    def append(self, item):
        self.data.append(item)

    def _get_data_attr(self, attribute):
        """Get attribute from internal data from a call of DataCollection.attribute"""
        return np.array([getattr(datum, attribute, np.nan) for datum in self.data])
    
    # Properties accessed from internal data --------------------------------- #
    @property
    def instrument(self):
        return self._get_data_attr("instrument")
    
    @property
    def cadence(self):
        return self._get_data_attr("cadence")
    
    @property
    def sector(self):
        return self._get_data_attr("sector")
    
    # Class methods ---------------------------------------------------------- #
    def stitch(self):
        t = np.hstack([datum.t for datum in self.data])
        y = np.hstack([datum.y for datum in self.data])
        e = np.hstack([datum.e for datum in self.data])
        return type(self.data[0])(*order_data(t, y, e), self.data[0].instrument)

# ---------------------------------------------------------------------------- #
# Helper functions                                                             #
# ---------------------------------------------------------------------------- #

def to_lightkurve(lc, **kwargs):
    try:
        from lightkurve import LightCurve as lk_LightCurve
    except ImportError:
        print('Lightkurve is not installed! Skipping construction')
        return lc

    return lk_LightCurve(time=lc.t, flux=lc.y, flux_err=lc.e, **kwargs)


def from_lightkurve(lc):
    t = lc.time.value  # in days
    y = lc.flux.value
    e = lc.flux_err.value if lc.flux_err is not None else np.full_like(y, np.std(y))
    return LightCurve(t, y, e)


# TODO: this was written by Claude - rewrite properly. although it works nicely
def bin_data(t, y, yerr=None, n_points=None, n_bins=None, t_bins=None, method='mean'):
    """
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
    """
    # ------------------------------------------------------------------------ #
    # Input validation                                                         #
    # ------------------------------------------------------------------------ #
    if yerr is None or sum(yerr) == 0:
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

    # ------------------------------------------------------------------------ #
    # Build a list of index slices, one per bin                                #
    # ------------------------------------------------------------------------ #
    slices = []  # each element is a boolean mask or array of indices

    if n_points is not None:
        # fixed number of data-points per bin -------------------------------- #
        n_points = int(n_points)
        if n_points < 1:
            raise ValueError("n_points must be >= 1.")
        starts = range(0, N, n_points)
        slices = [np.arange(s, min(s + n_points, N)) for s in starts]

    elif n_bins is not None:
        # fixed total number of (time-equal) bins ---------------------------- #
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
        # fixed time-span per bin -------------------------------------------- #
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

    # ------------------------------------------------------------------------ #
    # Compute bin statistics                                                   #
    # ------------------------------------------------------------------------ #
    t_out, y_out, yerr_out = [], [], []

    for idx in slices:
        t_bin   = t[idx]
        y_bin   = y[idx]
        err_bin = yerr[idx]
        n       = len(idx)

        if method == "mean":
            # Inverse-variance weighted mean
            if no_yerr:
                t_centre     = np.mean(t_bin)
                y_centre     = np.mean(y_bin)
                y_err_centre = np.std(y_bin) / np.sqrt(n)

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

    return np.array(t_out), np.array(y_out), np.array(yerr_out)


def order_data(t, y, e=None):
    '''
    Ensure a dataset is in order.

    Parameters
    ----------
    t : array-like

    TODO: make this able to sort any number of input arrays? as an input list? 
    '''
    idx = np.argsort(t)

    if e is None:
        return t[idx], y[idx], None
    return t[idx], y[idx], e[idx]


def fold_data(t, y, period, t0=None, e=None):
    if t0 is None:
        fold_t = (t % period) / period
    else:
        fold_t = ((t - t0 + (period/2)) % period) / period
    return order_data(fold_t, y, e)


def fold_data_alternate(t, y, period, t0=None, e=None):
    '''Slighty faster, but I stole it from transitleastsquares so I don't want to use it'''
    if t0 is None:
        fold_t = t / period - np.floor(t / period)
    else:
        fold_t = (t - t0) / period - np.floor((t - t0) / period)
    return order_data(fold_t, y, e)


def get_lightcurve(system_name, lc_directory, missions, authors={}, cadences='longest', selection='all', extract_ffi=False, fill_gaps=False,
                   overwrite=False, save_format='pickle', system=None, mask_transits=False, mask_tolerance=4):
    """
    Function to query and download lightcurves from space telescopes.
    The lightcurves are stored locally for easy retrieval, in a format of your choice.

    Parameters
    ----------
    system_name : string
        Name of the system to search
    lc_directory : string
        Filepath to save the lightcurves to, or to retrieve the lightcurves from.
    missions : string or list, {'Kepler', 'K2', 'TESS'}
        The space missions to pull data from.
    authors : dict, optional
        The author of the dataproduct to use for each mission.
        Defaults to 'Kepler' for Kepler, 'K2' for K2, and 'SPOC' for TESS.
    cadences : dict, string or int, optional
        The cadence/exposure time of each dataproduct to use.
        If a dict, it should be structured like {mission: {author: cadence}} to select cadences for products individually.
        If given an integer, all products with that cadence (in seconds) will be selected.
        If given a string, cadences will be selected for all authors by the following:
            - 'longest' or 'shortest' will select the longest or shortest cadences available.
            - 'long' selects 10-min and 30-min cadences.
            - 'short' selects 1-min and 2-min cadences.
            - 'fast' selects 20-sec cadences.
        By default, the longest cadences will be returned.
    selection : dict, string, list, or int, optional
        The indices of the lightcurves for each cadence to download.
        If a dict, it should be structured like {mission: {author: {cadence: selection}}} to select lightcurves for cadences individually.
            This selection can be a list or an integer.
        If a list or int, this will be used to select for all products.
        If 'all', all lightcurves will be returned.
        Defaults to 'all'.
    extract_ffi : bool or list, optional
        If True or a list of flux types and 'missions' contains 'TESS', eleanor will extract lightcurves
        from TESS Full Frame Images. These will be saved as additional lightcurves, separate from the others.
        If a list, it should contain the flux types to extract, e.g. ['corr', 'pca'].
        Valid flux types are 'raw', 'corr', 'pca', 'psf'. Defaults to ['corr', 'pca'] if True.
    fill_gaps : bool, optional
        Whether to only get FFI lightcurves for sectors missing existing lightcurves.
        Defaults to False.
    overwrite : bool, optional
        Whether to download lightcurves if a lightcurve for this system already exists in lc_directory.
        If false, the saved lightcurve will always be loaded, even if other kwargs are different.
    save_format : string, optional, {'pickle', 'csv', 'dat', 'txt', 'fits'}
        Format to save the lightcurves in.
    system : shelter.system, optional
        A shelter System() object.
        If provided, any attached planet objects can be used to mask transits prior to saving, to reduce file size.
        However, it is recommended to mask out transits after flattening, by using shelter's built-in
        Lightcurve.flatten().mask_transits() functions.
    mask_transits : bool, optional
        If True and a system object is provided, transits will be masked out of the lightcurves before saving.
    mask_tolerance : int or float, optional
        The tolerance for masking transits, in units of the transit duration. Defaults to 4.

    Returns
    -------
    lc : LightCurve or DataCollection object

    Raises
    ------
    ValueError
        If an unknown mission is provided.
    Exception
        If no search results are found for a given mission/author/cadence combination.
    """
    def mask_lightcurve_transits(lc, system, tolerance=2, only_mask=False):
        periods = []
        durations = []
        transit_times = []
        for planet in system.planets:
            periods.append(planet.period)
            durations.append(planet.duration / (24 / tolerance))
            transit_times.append(planet.time_of_midtransit)

        transit_mask = lc.create_transit_mask(periods, transit_times, durations)
        if only_mask:
            return transit_mask
        else:
            return lc[transit_mask]

    lc_filename = os.path.join(lc_directory, f"{system_name}_lightcurve.pkl")

    valid_missions = {'Kepler', 'K2', 'TESS'}
    invalid = set(missions) - valid_missions
    if invalid:
        raise ValueError(f"Unknown missions: {invalid}. Choose from {valid_missions}.")

    preferred_authors = {'Kepler': 'Kepler', 'K2': 'K2', 'TESS': 'SPOC'}
    sector_keys = {'Kepler': 'quarter', 'K2': 'campaign', 'TESS': 'sector'}

    # Normalise extract_ffi to a list of flux types, or False
    if extract_ffi is True:
        ffi_flux_types = ['corr', 'pca']
    elif isinstance(extract_ffi, list):
        ffi_flux_types = extract_ffi
        extract_ffi = True
    else:
        ffi_flux_types = []

    # Fill out input dictionaries ---------------------------------------- #
    def to_list(val):
        return [val] if isinstance(val, (str, int)) else list(val)

    def to_dict_over_missions(val, missions):
        return {m: val for m in missions} if not isinstance(val, dict) else val

    # Normalize missions
    missions = to_list(missions)

    # Normalize authors
    if not authors:
        authors = {m: preferred_authors[m] for m in missions}
    else:
        authors = to_dict_over_missions(to_list(authors), missions)
        authors = {m: to_list(authors[m]) if not isinstance(authors[m], list) else authors[m] for m in missions}

    # Normalize cadences
    if not isinstance(cadences, dict):
        cadences = {m: {a: to_list(cadences) for a in authors[m]} for m in missions}

    # Normalize selections
    if not isinstance(selection, dict):
        selection = {m: {a: {c: selection for c in cadences[m][a]} for a in authors[m]} for m in missions}

    # ------------------------------------------------------------------------ #
    # Check if file exists and read it if not forcing download                 #
    # ------------------------------------------------------------------------ #
    if os.path.exists(lc_filename) and not overwrite:
        print(f"Loading lightcurve data for {system_name} from local file.")
        if save_format == 'pickle':
            import pickle
            with open(lc_filename, 'rb') as f:
                data, mask, expt = pickle.load(f)

        if save_format == 'json':
            import json
            with open(lc_filename, 'r', encoding='utf-8') as f:
                data, mask, expt = json.load(f)

    # ------------------------------------------------------------------------ #
    # If it doesn't exist, download                                            #
    # ------------------------------------------------------------------------ #
    else:
        data = {}
        mask = {}
        expt = {}

        # -------------------------------------------------------------------- #
        # Loop over all MAST combinations                                      #
        # -------------------------------------------------------------------- #
        try:
            import lightkurve as lk
        except ImportError:
            print("Lightkurve is not installed! Skipping download")
        else:
            for mission in missions:
                for author in authors[mission]:
                    for cadence in cadences[mission][author]:
                        instrument = mission
                        select = selection[mission][author][cadence]
                        if len(authors) != 1:
                            instrument += ('-' + author)
                        if len(cadences) != 1:
                            instrument += ('-' + cadence)

                        data[instrument] = {}
                        mask[instrument] = {}

                        # Search for lightcurve data ------------------------- #
                        print(f"Downloading {cadence}-second lightcurve data by {author} from {mission} for {system_name}.")
                        search_result = lk.search_lightcurve(system_name, mission=mission, author=author, exptime=cadence)

                        print(search_result)
                        if len(search_result) == 0:
                            raise Exception('No search results found.')

                        # Handle 'longest' and 'shortest' cadence options ---- #
                        if cadence == 'longest':
                            cadence = max(search_result.exptime)
                        elif cadence == 'shortest':
                            cadence = min(search_result.exptime)
                        search_result = search_result[search_result.exptime == cadence]

                        # Download and stitch lightcurves -------------------- #
                        lc_collection = search_result.download_all()
                        for lc in lc_collection:
                            epoch = get_epoch(mission)
                            lc.time = lc.time + epoch  # Adjust time to absolute BJD

                        if select != 'all':
                            lc_collection = lc_collection[select]
                        if isinstance(select, int):
                            lc_collection = [lc_collection]

                        if np.any(lc_collection.flux) is None or np.any(lc_collection.flux_err) is None:
                            print("Warning: Downloaded lightcurve data has None values in flux or flux_err.")

                        for lc in lc_collection:
                            sector = getattr(lc, sector_keys[mission])
                            lc = lc.remove_nans()

                            # Mask transits ---------------------------------- #
                            if (system is not None) and mask_transits:
                                if len(system.planets) > 0:
                                    lc = mask_lightcurve_transits(lc, system, tolerance=mask_tolerance)

                                # If the whole sector is masked out, move on
                                if len(lc.t) == 0:
                                    continue

                            # Extract time and flux data --------------------- #
                            t = lc.time.value  # in days
                            y = lc.flux.value
                            e = lc.flux_err.value if lc.flux_err is not None else np.full_like(y, np.std(y))

                            # Convert data to plain NumPy arrays ------------- #
                            t_data = np.array(t.data if hasattr(t, 'data') else t)
                            t_mask = np.array(t.mask if hasattr(t, 'mask') else None)
                            y_data = np.array(y.data if hasattr(y, 'data') else y)
                            y_mask = np.array(y.mask if hasattr(y, 'mask') else None)
                            e_data = np.array(e.data if hasattr(e, 'data') else e)
                            e_mask = np.array(e.mask if hasattr(e, 'mask') else None)

                            # Store in dictionaries -------------------------- #
                            data[instrument][sector] = {'t': t_data, 'y': y_data, 'e': e_data}
                            mask[instrument][sector] = {'t': t_mask, 'y': y_mask, 'e': e_mask}
                        expt[instrument] = cadence

        # -------------------------------------------------------------------- #
        # Extract FFI lightcurves via eleanor                                  #
        # -------------------------------------------------------------------- #
        if extract_ffi and 'TESS' in missions:
            try:
                import eleanor
            except ImportError:
                print("eleanor is not installed. Skipping FFI extraction.")
            else:
                print(f"Extracting FFI lightcurves for {system_name} via eleanor.")

                # Resolve the target and find available sectors -------------- #
                try:
                    star = eleanor.Source(name=system_name, auto_submit=False)
                    sectors_available = star.sectors
                except Exception as exc:
                    print(f"eleanor could not resolve {system_name}: {exc}")
                    sectors_available = []

                epoch = get_epoch('TESS')

                # Exclude sectors available from MAST if requested ----------- #
                if fill_gaps:
                    sectors_downloaded = []
                    for instrument in data.keys():
                        for sector in instrument.keys():
                            sectors_downloaded.append(sector)
                    sectors_downloaded = set(sectors_downloaded)
                    sectors_available = [sector for sector in sectors_available if sector not in sectors_downloaded]

                for flux_type in ffi_flux_types:
                    instrument = f"TESS-FFI-{flux_type}"
                    data[instrument] = {}
                    mask[instrument] = {}

                    for sector in sectors_available:
                        try:
                            star_sector = eleanor.Source(
                                name=system_name, sector=sector, auto_submit=False
                            )
                            datum = eleanor.TargetData(
                                star_sector,
                                do_psf=(flux_type == 'psf'),
                                do_pca=(flux_type in ('pca', 'corr')),
                            )
                            datum.save()  # Cache locally so re-runs are fast

                            q = datum.quality_mask
                            t_raw = datum.time[q] + epoch  # BTJD → absolute BJD

                            flux_arr = getattr(datum, f"{flux_type}_flux")[q]
                            err_arr = (
                                datum.flux_err[q]
                                if datum.flux_err is not None
                                else np.full_like(flux_arr, np.nanstd(flux_arr))
                            )

                            # Normalise to median = 1 for consistency with MAST lcs
                            med = np.nanmedian(flux_arr)
                            if med != 0:
                                flux_arr = flux_arr / med
                                err_arr  = err_arr  / med

                            # Mask transits ---------------------------------- #
                            if system is not None and mask_transits and len(system.planets) > 0:
                                _tmp = LightCurve(t_raw, flux_arr, err_arr)
                                _transit_mask = mask_lightcurve_transits(
                                    _tmp, system, tolerance=mask_tolerance, only_mask=True
                                )
                                t_raw    = t_raw[_transit_mask]
                                flux_arr = flux_arr[_transit_mask]
                                err_arr  = err_arr[_transit_mask]

                                if len(t_raw) == 0:
                                    continue

                            data[instrument][sector] = {
                                't': np.array(t_raw),
                                'y': np.array(flux_arr),
                                'e': np.array(err_arr),
                            }
                            mask[instrument][sector] = {'t': None, 'y': None, 'e': None}

                        except Exception as exc:
                            print(f"eleanor sector {sector} ({flux_type}) failed: {exc}")
                            continue

                    # Use datum.cadence if available, otherwise infer from sector
                    try:
                        expt[instrument] = int(datum.cadence)
                    except Exception:
                        expt[instrument] = 1800  # Conservative fallback: 30-min cadence

        # -------------------------------------------------------------------- #
        # Save to disk                                                         #
        # -------------------------------------------------------------------- #
        if save_format == 'pickle':
            import pickle
            with open(lc_filename, 'wb') as f:
                pickle.dump([data, mask, expt], f)

        if save_format == 'json':
            import json
            with open(lc_filename, 'w', encoding='utf-8') as f:
                json.dump([data, mask, expt], f, ensure_ascii=False, indent=4)

        print(f"All lightcurve data for {system_name} saved locally.")

    # ------------------------------------------------------------------------ #
    # Load data into LightCurve or DataCollection objects                      #
    # ------------------------------------------------------------------------ #
    collections = DataCollection()
    for key in data.keys():
        instrument_data = data.get(key, {})
        instrument_mask = mask.get(key, {})
        instrument_expt = expt.get(key, None)

        lcs = DataCollection()

        for sector in instrument_data.keys():
            sector_data = instrument_data[sector]
            sector_mask = instrument_mask[sector]

            # Recreate masked arrays ----------------------------------------- #
            t = np.ma.MaskedArray(data=sector_data['t'], mask=sector_mask['t'] if sector_mask['t'] is not None else False)
            y = np.ma.MaskedArray(data=sector_data['y'], mask=sector_mask['y'] if sector_mask['y'] is not None else False)
            e = np.ma.MaskedArray(data=sector_data['e'], mask=sector_mask['e'] if sector_mask['e'] is not None else False)

            lcs.append(LightCurve(t, y, e, instrument=key, cadence=instrument_expt))

        collections.append(lcs)

    if len(collections) == 1:
        return collections[0]
    if len(collections) == 0:
        print("No lightcurves found?")
        return None
    return collections
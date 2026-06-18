import os.path
import pickle
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
                y_flat = wotan.flatten(self.t, self.y, window=wind)
                return_obj.add_data(LightCurve(self.t, y_flat, self.e, self.instrument, self.cadence))

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
            return type(self)(self.data[idx])
        
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

'''
# TODO: rewrite this to be better - more flexible. harder, better, faster, stronger
def get_lightcurve(system, missions, lc_directory, cadences=None, overwrite=False,
                   selection='all', authors='SPOC', flatten=True, flatten_mask=False, only_transits=True, tolerance=4,
                   plots='none'):
    """ Function to retrieve lightcurve data for a system from Kepler or TESS, with local storage """

    def align_mission_times(lc_collection):
        """
        Automatically aligns times for multiple light curves from different missions.
        If multiple light curves are from the same mission and quarter, all light curves 
        with the shortest cadence are retained.
        
        Parameters:
        - lcs: list of LightCurve objects
        """    
        reverseauthors = {'Kepler': 'Kepler', 'K2': 'K2', 'SPOC': 'TESS', 'QLP': 'TESS'}

        # Group light curves by mission and quarter
        def get_mission_quarter(lc):
            mission = reverseauthors[lc.meta['AUTHOR']]
            quarter = lc.meta.get('QUARTER', lc.meta.get('SECTOR', None))  # Use QUARTER for Kepler/K2, SECTOR for TESS
            return (mission, quarter)
        
        grouped_lcs = {}
        for lc in lc_collection:
            mission_quarter = get_mission_quarter(lc)
            if mission_quarter not in grouped_lcs:
                grouped_lcs[mission_quarter] = []
            grouped_lcs[mission_quarter].append(lc)
        
        # Select all shortest-cadence light curves from each group
        filtered_lcs = []
        for mission_quarter, group in grouped_lcs.items():
            # Find the shortest cadence
            shortest_cadence = min(lc.meta['TIMEDEL'] for lc in group)  # TIMEDEL indicates cadence
            # Include all light curves with the shortest cadence
            shortest_cadence_lcs = [lc for lc in group if lc.meta['TIMEDEL'] == shortest_cadence]
            filtered_lcs.extend(shortest_cadence_lcs)
        
        # Align the times
        for lc in filtered_lcs:
            mission = reverseauthors[lc.meta['AUTHOR']]
            epoch = get_epoch(mission)
            lc.time = lc.time + epoch  # Adjust time to absolute BJD
        
        return lk.LightCurveCollection(filtered_lcs)
    
    def clean_lightcurve(lc_collection, system, selection=None, flatten=True, exptime=None, flatten_mask=False):
        """ Function to stitch together and reduce a collection of lightcurves """

        # Combine all lightcurve data into one object
        if selection == 'all':
            lc = lc_collection.stitch()
        elif isinstance(selection, int):
            lc = lc_collection[selection]
            exptime = exptime[selection]
        else:
            lc = lc_collection[selection].stitch()
            exptime = exptime[selection]

        # Remove NaNs
        lc = lc.remove_nans()

        # Flatten for trend correction
        if flatten:
            transit_duration = np.max([planet.duration for planet in system.planets]) * (3600 * 24)
            cadence = np.min(exptime.value)
            window = int((transit_duration / cadence) * 4)

            if window % 2 == 0:
                window = window + 1

            if flatten_mask:
                transit_mask = mask_lightcurve_transits(lc, system, tolerance=1.1, only_mask=True)
            else:
                transit_mask = None
            lc = lc.flatten(window_length=window, mask=transit_mask)
        
        return lc

    def mask_lightcurve_transits(lc, system, tolerance=2, only_mask=False):
        periods = []
        durations = []
        transit_times = []
        for planet in system.planets:
            periods.append(planet.period)
            durations.append(planet.duration/(24/tolerance))
            transit_times.append(planet.time_of_midtransit)

        transit_mask = lc.create_transit_mask(periods, transit_times, durations)
        if only_mask:
            return transit_mask
        else:
            return lc[transit_mask]

    lc_filename = os.path.join(lc_directory, f"{system.name}_lightcurve.pkl")

    # Initialize storage dictionaries
    times = {}
    fluxes = {}
    flux_errors = {}
    times_mask = {}
    fluxes_mask = {}
    flux_errors_mask = {}
    instruments = []

    # Define default cadences if not provided
    if cadences is None:
        cadences = {mission: 'longest' for mission in missions}

    if isinstance(cadences, str):
        cadences = {mission: cadences for mission in missions}

    if isinstance(authors, str):
        authors = [authors]

    for mission in missions:
        if isinstance(cadences[mission], str):
            cadences[mission] = [cadences[mission]]

        for cadence in cadences[mission]:
            for i, author in enumerate(authors):
                if isinstance(cadence, int):
                    instrument = mission + cadence
                else:
                    instrument = mission
                if len(authors) > 1:
                    instrument += f'-{author}'
                    selection_i = selection[i]
                else:
                    selection_i = selection
                instruments.append(instrument)
    
    # Check if file exists and read it if not forcing download
    if os.path.exists(lc_filename) and not overwrite:
        try:
            print(f"Loading lightcurve data for {system.name} from local file.")
            with open(lc_filename, 'rb') as f:
                stored_data, stored_masks = pickle.load(f)
                times = stored_data.get('times', {})
                fluxes = stored_data.get('fluxes', {})
                flux_errors = stored_data.get('flux_errors', {})
                times_mask = stored_masks.get('times_mask', {})
                fluxes_mask = stored_masks.get('fluxes_mask', {})
                flux_errors_mask = stored_masks.get('flux_errors_mask', {})

            for mission in missions:
                for cadence in cadences[mission]:
                    if isinstance(cadence, int):
                        instrument = mission + cadence
                    else:
                        instrument = mission

                    time_data, flux_data, flux_err_data = times[instrument], fluxes[instrument], flux_errors[instrument]
                    time_mask, flux_mask, flux_err_mask = times_mask[instrument], fluxes_mask[instrument], flux_errors_mask[instrument]

                    # Recreate masked arrays
                    time = np.ma.MaskedArray(data=time_data, mask=time_mask if time_mask is not None else False)
                    flux = np.ma.MaskedArray(data=flux_data, mask=flux_mask if flux_mask is not None else False)
                    flux_err = np.ma.MaskedArray(data=flux_err_data, mask=flux_err_mask if flux_err_mask is not None else False)

                    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)

                    # Put data into dictionaries
                    times[instrument], fluxes[instrument], flux_errors[instrument] = time_data, flux_data, flux_err_data
                    times_mask[instrument], fluxes_mask[instrument], flux_errors_mask[instrument] = time_mask, flux_mask, flux_err_mask
                    
            return times, fluxes, flux_errors, instruments
        
        except Exception:
            print(f"Error loading {cadence}-second {mission} lightcurve for {system.name} from local file.")

    for mission in missions:
        for cadence in cadences[mission]:
            for i, author in enumerate(authors):
                if isinstance(cadence, int):
                    instrument = mission + cadence
                else:
                    instrument = mission
                if len(authors) > 1:
                    instrument += f'-{author}'
                    selection_i = selection[i]
                else:
                    selection_i = selection
                try:
                    # Search for and download lightcurve data using Lightkurve
                    print(f"Downloading {cadence}-second lightcurve data from {mission} for {system.name}.")
                    search_result = lk.search_lightcurve(system.name, mission=mission, author=author, exptime=cadence)

                    # Handle 'longest' and 'shortest' cadence options
                    if cadence == 'longest':
                        max_cadence = max(search_result.exptime)
                        if max_cadence in [1800, 600]:
                            search_result = search_result[(search_result.exptime == 1800) | (search_result.exptime == 600)]
                        elif max_cadence in [120, 60]:
                            search_result = search_result[(search_result.exptime == 120) | (search_result.exptime == 60)]
                        else:
                            search_result = search_result[search_result.exptime == max_cadence]
                    elif cadence == 'shortest':
                        min_cadence = min(search_result.exptime)
                        if min_cadence in [60, 120]:
                            search_result = search_result[(search_result.exptime == 60) | (search_result.exptime == 120)]
                        elif min_cadence == 20:
                            search_result = search_result[search_result.exptime == 20]
                        else:
                            search_result = search_result[search_result.exptime == min_cadence]

                    print(search_result)
                    if len(search_result) == 0:
                        raise Exception('No search results found')
                    exptime = search_result.exptime

                    # Download & filter lightcurves if downloading from multiple missions
                    lc_collection = align_mission_times(search_result.download_all())
                    
                    # Combine and clean up lightcurve
                    lc = clean_lightcurve(lc_collection, system, selection=selection_i, flatten=flatten, exptime=exptime, flatten_mask=flatten_mask)

                    if lc.flux is None or lc.flux_err is None:
                        print("Warning: Downloaded lightcurve data has None values in flux or flux_err.")
                    
                    # Extract time and flux data
                    time = lc.time.value  # in days
                    flux = lc.flux.value
                    flux_err = lc.flux_err.value if lc.flux_err is not None else np.full_like(flux, 1e-4)

                    if only_transits:
                            lc = mask_lightcurve_transits(lc, system, tolerance=tolerance)
                            time = lc.time.value  # in days
                            flux = lc.flux.value
                            flux_err = lc.flux_err.value if lc.flux_err is not None else np.full_like(flux, 1e-4)

                    # Convert data to plain NumPy arrays to avoid memoryview issues
                    time_data = np.array(time.data if hasattr(time, 'data') else time)
                    time_mask = np.array(time.mask if hasattr(time, 'mask') else None)
                    flux_data = np.array(flux.data if hasattr(flux, 'data') else flux)
                    flux_mask = np.array(flux.mask if hasattr(flux, 'mask') else None)
                    flux_err_data = np.array(flux_err.data if hasattr(flux_err, 'data') else flux_err)
                    flux_err_mask = np.array(flux_err.mask if hasattr(flux_err, 'mask') else None)

                    # Put data into dictionaries
                    times[instrument], fluxes[instrument], flux_errors[instrument] = time_data, flux_data, flux_err_data
                    times_mask[instrument], fluxes_mask[instrument], flux_errors_mask[instrument] = time_mask, flux_mask, flux_err_mask
                    
                except Exception as e:
                    print(f"Error retrieving {cadence}-second {mission} lightcurve for {system.name}: {e}")
                    instruments.remove(instrument)
                    traceback.print_exc()
    
    # Save all mission data into a single pickle file
    with open(lc_filename, 'wb') as f:
        pickle.dump([{'times': times, 'fluxes': fluxes, 'flux_errors': flux_errors},
                        {'times_mask': times_mask, 'fluxes_mask': fluxes_mask, 'flux_errors_mask': flux_errors_mask}], f)
    print(f"All lightcurve data for {system.name} saved locally.")
    
    return times, fluxes, flux_errors, instruments
'''

def get_lightcurve(system_name, lc_directory, missions, authors={}, cadences='longest', selection='all',
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
    overwrite : bool, optional
        Whether to download lightcurves if a lightcurve for this system already exists in lc_directory.
        If false, the saved lightcurve will always be loaded, even if other kwargs are different.
    save_format : string, optional, {'pickle', 'csv', 'dat', 'txt', 'fits'}
        Format to save the 
    system : shelter.system, optional
        A shelter System() object.
        If provided, any attached planet objects can be used to mask transits prior to saving, to reduce file size.
        However, it is recommended to mask out transits after flattening, by using shelter's built-in Lightcurve.flatten().mask_transits() functions.

    Returns
    -------
    lc : LightCurve or DataCollection object

    Raises
    ------
    """
    def mask_lightcurve_transits(lc, system, tolerance=2, only_mask=False):
        periods = []
        durations = []
        transit_times = []
        for planet in system.planets:
            periods.append(planet.period)
            durations.append(planet.duration/(24/tolerance))
            transit_times.append(planet.time_of_midtransit)

        transit_mask = lc.create_transit_mask(periods, transit_times, durations)
        if only_mask:
            return transit_mask
        else:
            return lc[transit_mask]

    lc_filename = os.path.join(lc_directory, f"{system_name}_lightcurve.pkl")

    # ------------------------------------------------------------------------ #
    # Check if file exists and read it if not forcing download                 #
    # ------------------------------------------------------------------------ #
    if os.path.exists(lc_filename) and not overwrite:
        print(f"Loading lightcurve data for {system_name} from local file.")
        with open(lc_filename, 'rb') as f:
            data, mask, expt = pickle.load(f)
    # ------------------------------------------------------------------------ #
    # If it doesn't exist, download                                            #
    # ------------------------------------------------------------------------ #
    else:
        try:
            import lightkurve as lk
        except ImportError:
            print("Lightkurve is not installed! Skipping download")
            return None
        
        preferred_authors = {'Kepler': 'Kepler', 'K2': 'K2', 'TESS': 'SPOC'}

        # Fill out input dictionaries ---------------------------------------- #
        if isinstance(missions, str):
            missions = [missions]

        if len(authors) == 0:
            authors = {mission: preferred_authors[mission] for mission in missions}
        if isinstance(authors, str):
            authors = {mission: authors for mission in missions}

        if isinstance(cadences, str) or isinstance(cadences, int):
            cadences = {mission: {author: cadences for author in authors[mission]} for mission in missions}

        if isinstance(selection, str) or isinstance(selection, int) or isinstance(selection, list):
            selection = {mission: {author: {cadence: selection for cadence in cadences[mission][author]} for author in authors[mission]} for mission in missions}

        data = {}
        mask = {}
        expt = {}

        # -------------------------------------------------------------------- #
        # Loop over all combinations                                           #
        # -------------------------------------------------------------------- #
        for mission in missions:
            for author in authors[mission]:
                for cadence in cadences[author][mission]:
                    instrument = mission
                    if len(authors) != 1:
                        instrument += ('-' + author)
                    if len(cadences) != 1:
                        instrument += ('-' + cadence)

                    # Search for lightcurve data ----------------------------- #
                    print(f"Downloading {cadence}-second lightcurve data by {author} from {mission} for {system_name}.")
                    search_result = lk.search_lightcurve(system_name, mission=mission, author=author, exptime=cadence)

                    print(search_result)
                    if len(search_result) == 0:
                        raise Exception('No search results found.')

                    # Handle 'longest' and 'shortest' cadence options -------- #
                    if cadence == 'longest':
                        cadence = max(search_result.exptime)
                    elif cadence == 'shortest':
                        cadence = min(search_result.exptime)
                    search_result = search_result[search_result.exptime == cadence]

                    # Download and stitch lightcurves ------------------------ #
                    lc_collection = search_result.download_all()
                    for lc in lc_collection:
                        epoch = get_epoch(mission)
                        lc.time = lc.time + epoch  # Adjust time to absolute BJD

                    if selection == 'all':
                        lc = lc_collection.stitch()
                    elif isinstance(selection, int):
                        lc = lc_collection[selection]
                        exptime = exptime[selection]
                    else:
                        lc = lc_collection[selection].stitch()
                        exptime = exptime[selection]

                    # Remove NaNs
                    lc = lc.remove_nans()

                    # Mask transits ------------------------------------------ #
                    if (system is not None) and mask_transits:
                        if len(system.planets) > 0:
                            lc = mask_lightcurve_transits(lc, system, mask_tolerance=mask_tolerance)

                    if lc.flux is None or lc.flux_err is None:
                        print("Warning: Downloaded lightcurve data has None values in flux or flux_err.")

                    # Extract time and flux data ----------------------------- #
                    t = lc.time.value  # in days
                    y = lc.flux.value
                    e = lc.flux_err.value if lc.flux_err is not None else np.full_like(y, np.std(y))

                    # Convert data to plain NumPy arrays --------------------- #
                    t_data = np.array(t.data if hasattr(t, 'data') else t)
                    t_mask = np.array(t.mask if hasattr(t, 'mask') else None)
                    y_data = np.array(y.data if hasattr(y, 'data') else y)
                    y_mask = np.array(y.mask if hasattr(y, 'mask') else None)
                    e_data = np.array(e.data if hasattr(e, 'data') else e)
                    e_mask = np.array(e.mask if hasattr(e, 'mask') else None)

                    # Store in dictionaries ---------------------------------- #
                    data[instrument] = {'t': t_data, 'y': y_data, 'e': e_data}
                    mask[instrument] = {'t': t_mask, 'y': y_mask, 'e': e_mask}
                    expt[instrument] = cadence
        
        # Save all mission data into a single pickle file -------------------- #
        with open(lc_filename, 'wb') as f:
            pickle.dump([data, mask, expt], f)
        print(f"All lightcurve data for {system_name} saved locally.")

    # ------------------------------------------------------------------------ #
    # Load data into LightCurve or DataCollection objects                      #
    # ------------------------------------------------------------------------ #
    lcs = []
    for key in data.keys():
        instrument_data = data.get(key, {})
        instrument_mask = mask.get(key, {})
        instrument_expt = expt.get(key, None)

        # Recreate masked arrays --------------------------------------------- #
        t = np.ma.MaskedArray(data=instrument_data['t'], mask=instrument_mask['t'] if instrument_mask['t'] is not None else False)
        y = np.ma.MaskedArray(data=instrument_data['y'], mask=instrument_mask['y'] if instrument_mask['y'] is not None else False)
        e = np.ma.MaskedArray(data=instrument_data['e'], mask=instrument_mask['e'] if instrument_mask['e'] is not None else False)

        lcs.append(LightCurve(t, y, e, instrument=key, cadence=instrument_expt))
    
    if len(lcs) == 1:
        return lcs[0]
    if len(lcs) == 0:
        print("No lightcurves found?")
        return None
    return DataCollection(lcs)
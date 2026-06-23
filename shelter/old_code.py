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


def get_lightcurve(system_name, lc_directory, missions, authors={}, cadences='longest', selection='all', extract_ffi=False,
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
    extract_ffi : bool, optional
        If True and 'missions' contains 'TESS', ELEANOR will get lightcurves from TESS Full Frame Images.
        These will be saved as an additional lightcurve, separate from the others
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
        import pickle
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
        
        preferred_authors = {'Kepler': ['Kepler'], 'K2': ['K2'], 'TESS': ['SPOC']}

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

        data = {}
        mask = {}
        expt = {}

        # -------------------------------------------------------------------- #
        # Loop over all combinations                                           #
        # -------------------------------------------------------------------- #
        for mission in missions:
            for author in authors[mission]:
                for cadence in cadences[mission][author]:
                    instrument = mission
                    select = selection[mission][author][cadence]
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

                    if select == 'all':
                        lc = lc_collection.stitch()
                    elif isinstance(select, int):
                        lc = lc_collection[select]
                    else:
                        lc = lc_collection[select].stitch()

                    # Remove NaNs
                    lc = lc.remove_nans()

                    # Mask transits ------------------------------------------ #
                    if (system is not None) and mask_transits:
                        if len(system.planets) > 0:
                            lc = mask_lightcurve_transits(lc, system, tolerance=mask_tolerance)

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
        import pickle
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


def get_lightcurve(system_name, lc_directory, missions, authors={}, cadences='longest', selection='all', extract_ffi=False,
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
    selections : dict, string, list, or int, optional
        The indices of the lightcurves for each cadence to download.
        If a dict, it should be structured like {mission: {author: {cadence: selection}}} to select lightcurves for cadences individually.
            This selection can be a list or an integer.
        If a list or int, this will be used to select for all products.
        If 'all', all lightcurves will be returned.
        Defaults to 'all'.
    extract_ffi : bool, optional
        If True and 'missions' contains 'TESS', ELEANOR will get lightcurves from TESS Full Frame Images.
        These will be saved as an additional lightcurve, separate from the others
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

    valid_missions = {'Kepler', 'K2', 'TESS'}
    invalid = set(missions) - valid_missions
    if invalid:
        raise ValueError(f"Unknown missions: {invalid}. Choose from {valid_missions}.")

    preferred_authors = {'Kepler': 'Kepler', 'K2': 'K2', 'TESS': 'SPOC'}
    sector_keys = {'Kepler': 'quarter', 'K2': 'campaign', 'TESS': 'sector'}

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
            # Load mission data from pickle file ----------------------------- #
            with open(lc_filename, 'rb') as f:
                data, mask, expt = pickle.load(f)
        
        if save_format == 'json':
            import json
            # Load mission data from json file ------------------------------- #
            with open(lc_filename, 'r', encoding='utf-8') as f:
                data, mask, expt = json.load(f)
    # ------------------------------------------------------------------------ #
    # If it doesn't exist, download                                            #
    # ------------------------------------------------------------------------ #
    else:
        try:
            import lightkurve as lk
        except ImportError:
            print("Lightkurve is not installed! Skipping download")
            return None

        data = {}
        mask = {}
        expt = {}

        # -------------------------------------------------------------------- #
        # Loop over all combinations                                           #
        # -------------------------------------------------------------------- #
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

                    if select != 'all':
                        lc_collection = lc_collection[select]
                    if isinstance(select, int):
                        lc_collection = [lc_collection]

                    if np.any(lc_collection.flux) is None or np.any(lc_collection.flux_err) is None:
                        print("Warning: Downloaded lightcurve data has None values in flux or flux_err.")

                    for lc in lc_collection:
                        sector = getattr(lc, sector_keys[mission])
                        lc = lc.remove_nans()

                        # Mask transits -------------------------------------- #
                        if (system is not None) and mask_transits:
                            if len(system.planets) > 0:
                                lc = mask_lightcurve_transits(lc, system, tolerance=mask_tolerance)

                            # If the whole sector is masked out, move on
                            if len(lc.t) == 0:
                                continue

                        # Extract time and flux data ------------------------- #
                        t = lc.time.value  # in days
                        y = lc.flux.value
                        e = lc.flux_err.value if lc.flux_err is not None else np.full_like(y, np.std(y))

                        # Convert data to plain NumPy arrays ----------------- #
                        t_data = np.array(t.data if hasattr(t, 'data') else t)
                        t_mask = np.array(t.mask if hasattr(t, 'mask') else None)
                        y_data = np.array(y.data if hasattr(y, 'data') else y)
                        y_mask = np.array(y.mask if hasattr(y, 'mask') else None)
                        e_data = np.array(e.data if hasattr(e, 'data') else e)
                        e_mask = np.array(e.mask if hasattr(e, 'mask') else None)

                        # Store in dictionaries ------------------------------ #
                        data[instrument][sector] = {'t': t_data, 'y': y_data, 'e': e_data}
                        mask[instrument][sector] = {'t': t_mask, 'y': y_mask, 'e': e_mask}
                    expt[instrument] = cadence
        
        if save_format == 'pickle':
            import pickle
            # Save all mission data into a single pickle file ---------------- #
            with open(lc_filename, 'wb') as f:
                pickle.dump([data, mask, expt], f)

        if save_format == 'json':
            import json
            # Save all mission data into a single json file ------------------ #
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

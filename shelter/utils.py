import inspect
import numpy as np
import astropy.constants as const
import astropy.units as u

'''def extract_kwargs(function, kwargs):
    args = function.__init__.__code__.co_varnames
    new_args = {}

    for arg in args:
        if arg in kwargs:
            new_args[arg] = kwargs[arg]

    return new_args'''

def extract_kwargs(function, kwargs):
    valid = inspect.signature(function).parameters
    return {k: v for k, v in kwargs.items() if k in valid}

def to_list_of_arrays(x):
    """Normalise input to a list of arrays."""
    if isinstance(x, (list, tuple)):
        return x
    return [x]

def get_epoch(mission):
    epochs = {'Kepler': 2454833, 'K2': 2454833, 'TESS': 2457000}
    try:
        return epochs[mission]
    except KeyError:
        print(f'{mission} is not a supported mission!')
        return None

def get_alias_mask(periods, period, threshold=0.1, order=[1/4, 1/3, 1/2, 1, 2, 3, 4]):
    mask = np.ones(len(periods)) * np.bool(False)

    for l in order:
        alias = period * l
        mask_l = np.logical_and(periods >= (alias - threshold), periods <= (alias + threshold))
        mask = np.logical_or(mask, mask_l)

    return mask

# ---------------------------------------------------------------------------- #
# Warnings                                                                     #
# ---------------------------------------------------------------------------- #

class DataWarning(Warning):
    pass

# ---------------------------------------------------------------------------- #
# Transit Relations                                                            #
# ---------------------------------------------------------------------------- #

def distance_by_irradiation(radius, temperature, flux=4):
    '''Calculate the distance at which a planet would receive the same flux as Earth, given the star's radius and temperature.
    Parameters:
    radius (float): Radius of the star in solar radii.
    temperature (float): Effective temperature of the star in Kelvin.
    flux (float): Target irradiation in Earth irradiance (default is 4)'''
    earth_flux = const.L_sun.value / (4 * np.pi * (1 * u.au.to(u.m))**2)

    def luminosity_from_star(radius, temperature):
        return 4 * np.pi * (radius * u.R_sun.to(u.m))**2 * const.sigma_sb.value * (temperature)**4

    def distance_from_flux(luminosity, flux):
        return np.sqrt(luminosity / (4 * np.pi * flux))

    luminosity = luminosity_from_star(radius, temperature)
    distance = distance_from_flux(luminosity, flux * earth_flux)
    return distance / u.au.to(u.m)

def transit_depth(radius, star_radius):
    '''Calculate the transit depth for a planet'''
    return (radius / star_radius)**2

def scaled_radius(depth, u_depth=None):
    '''Calculate the scaled radius from transit depth'''
    rp_rs = depth**0.5
    if u_depth is not None:
        u_rp_rs = 0.5 * (u_depth * rp_rs) / depth
        return rp_rs, u_rp_rs
    return rp_rs

def keplers_third_law(distance, star_mass):
    '''Calculate the orbital period in days using Kepler's Third Law.
    Parameters:
    distance (float): Orbital distance in AU.
    star_mass (float): Mass of the star in solar masses.'''
    G = const.G.value  # Gravitational constant in m^3 kg^-1 s^-2
    M_sun = const.M_sun.value  # Solar mass in kg
    AU = u.au.to(u.m)  # Astronomical unit in meters

    # Convert distance from AU to meters and star mass from solar masses to kg
    a = distance * AU
    M = star_mass * M_sun

    # Calculate orbital period in seconds using Kepler's Third Law
    P_seconds = 2 * np.pi * np.sqrt(a**3 / (G * M))

    # Convert orbital period from seconds to days
    P_days = P_seconds / (3600 * 24)
    return P_days

def transit_duration(period, semimajor_axis, star_radius, depth):
    '''Calculate the transit duration in hours using a rearranged version of the transit duration formula.
    Parameters:
    period (float): Orbital period in days.
    semimajor_axis (float): Semi-major axis of the planet's orbit in AU.
    star_radius (float): Radius of the star in solar radii.
    depth (float): Transit depth in ppm.'''
    # Convert inputs to consistent units
    P = period * 24 * 3600  # Convert period from days to seconds
    a = semimajor_axis * u.au.to(u.m)  # Convert semi-major axis from AU to meters
    R_star = star_radius * u.R_sun.to(u.m)  # Convert star radius from solar radii to meters
    delta = depth / 1e6  # Convert depth from ppm to fraction

    # Calculate transit duration using the rearranged formula
    W = (P / np.pi) * np.arcsin((R_star / a) * np.sqrt((1 + np.sqrt(delta))**2 - (R_star / a)**2))
    
    return W / 3600  # Convert transit duration from seconds to hours

def transit_SNR(depth, duration, noise, n_transits):
    return (depth / noise) * ((duration * 3600) / n_transits)**0.5
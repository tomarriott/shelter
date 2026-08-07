import astropy.units as u

################################################################################
# - DICTIONARIES ------------------------------------------------------------- #
################################################################################

exoarchive_planet_params = {
    "semiamplitude":        "pl_rvamp",
    "semimajor_axis":       "pl_orbsmax",
    "scaled_semimajor_axis":"pl_ratdor",
    "period":               "pl_orbper",
    "mass":                 "pl_bmasse",
    "radius":               "pl_rade",
    "scaled_radius":        "pl_ratror",
    "density":              "pl_dens",
    "eccentricity":         "pl_orbeccen",
    "inclination":          "pl_orbincl",
    "insolation_flux":      "pl_insol",
    "time_of_midtransit":   "pl_tranmid",
    "temperature":          "pl_eqt",
    "impact_parameter":     "pl_imppar",
    "depth":                "pl_trandep",
    "duration":             "pl_trandur",
    "arg_periastron":       "pl_orblper",
    "time_periastron":      "pl_orbtper",
    "star_temperature":     "st_teff",
    "star_mass":            "st_mass",
    "star_radius":          "st_rad",
    "star_density":         "st_dens",
    "ra":                   "ra",
    "dec":                  "dec",
    "distance":             "sy_dist",
}

exoarchive_planet_names = {
    "star_name":            "hostname",
    "name":                 "pl_name",
    "letter":               "pl_letter",
    "time_ref":             "pl_tsystemref",
    "publication":          "pl_refname",
    "disposition":          "disposition",
}

exoarchive_star_params = {
    "temperature":          "st_teff",
    "radius":               "st_rad",
    "mass":                 "st_mass",
    "density":              "st_dens",
    "metallicity":          "st_met",
    "luminosity":           "st_lum",
    "ra":                   "ra",
    "dec":                  "dec",
    "distance":             "sy_dist",
}

exoarchive_star_names = {
    "name":                 "hostname",
    "rastr":                "rastr",
    "decstr":               "decstr", 
    "metallicity_ratio":    "st_metratio",
    "spectral_type":        "st_spectype",
}

planet_aliases = {
    "semiamplitude":        ["K"],
    "period":               ["P", "per"],
    "time_of_midtransit":   ["t0", "t_0", "midtransit_time", "transit_time", "time_of_transit"],
    "semimajor_axis":       ["a", "sma"],
    "scaled_semimajor_axis":["a_R", "ssma"],
    "mass":                 ["m"],
    "radius":               ["r"],
    "scaled_radius":        ["r_R", "p"],
    "density":              ["rho", "dens"],
    "eccentricity":         ["e", "ecc"],
    "inclination":          ["i", "inc"],
    "insolation_flux":      ["S", "insol"],
    "temperature":          ["T_eq"],
    "impact_parameter":     ["b", "imppar", "imp"],
    "depth":                ["delta", "D"],
    "duration":             ["W", "dur", "T_14"],
    "arg_periastron":       ["omega", "w"],
    "time_periastron":      ["t_p"],
    "star_temperature":     ["T_star"],
    "star_mass":            ["M_star"],
    "star_radius":          ["R_star"],
    "star_density":         ["rho_star"],
    "ra":                   ["RA"],
    "dec":                  ["DEC"],
    "distance":             ["dist"], 
    "dispostion":           ["disp"],
}

star_aliases = {
    "temperature":          ["T", "T_star"],
    "radius":               ["R", "R_star"],
    "mass":                 ["M", "M_star"],
    "density":              ["rho", "rho_star"],
    "metallicity":          ["met"],
    "luminosity":           ["L", "L_star"],
    "ra":                   ["RA"],
    "dec":                  ["DEC"],
    "distance":             ["dist"],
    "spectral_type":        ["SpType"],
    "metallicity_ratio":    ["metratio"],
}

# Define uncertainty suffix aliases
uncertainty_suffixes = {
    "_upper":               ["_u", "err1"],
    "_lower":               ["_l", "err2"],
}

exoarchive_uncertainties = {
    "upper":                "err1",
    "lower":                "err2",
}

# Define units
planet_units = {
    "semiamplitude":        u.m/u.s,
    "semimajor_axis":       u.AU,
    "scaled_semimajor_axis":u.dimensionless_unscaled,
    "period":               u.day,
    "mass":                 u.M_earth,
    "radius":               u.R_earth,
    "scaled_radius":        u.dimensionless_unscaled,
    "density":              u.kg/(u.m**3),
    "eccentricity":         u.dimensionless_unscaled,
    "inclination":          u.degree,
    "insolation_flux":      u.W,
    "time_of_midtransit":   u.day,
    "temperature":          u.K,
    "impact_parameter":     u.dimensionless_unscaled,
    "depth":                u.percent,
    "duration":             u.hour,
    "arg_periastron":       u.degree,
    "time_periastron":      u.day,
    "star_temperature":     u.K,
    "star_mass":            u.M_sun,
    "star_radius":          u.R_sun,
    "star_density":         u.kg/(u.m**3),
    "ra":                   u.degree,
    "dec":                  u.degree,
    "distance":             u.parsec,
}

star_units = {
    "temperature":          u.K,
    "radius":               u.R_sun,
    "mass":                 u.M_sun,
    "density":              u.kg/(u.m**3),
    "metallicity":          u.dimensionless_unscaled,
    "luminosity":           u.L_sun,
    "ra":                   u.degree,
    "dec":                  u.degree,
    "distance":             u.parsec,
}
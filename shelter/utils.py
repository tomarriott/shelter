import inspect
import numpy as np

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
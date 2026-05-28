def extract_kwargs(function, kwargs):
    args = function.__init__.__code__.co_varnames
    new_args = {}

    for arg in args:
        if arg in kwargs:
            new_args[arg] = kwargs[arg]

    return new_args
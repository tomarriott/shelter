try:
    from astropy.timeseries import BoxLeastSquares
    BLS_active = True
except ImportError:
    print('Box Least Squares is not installed! The search_BLS class will not be available.')
    BLS_active = False

try:
    from transitleastsquares import transitleastsquares
    TLS_active = True
except ImportError:
    print('Transit Least Squares is not installed! The search_TLS class will not be available.')
    TLS_active = False

try:
    from transitleastsquares import transitleastsquares
    GTLS_active = True
except ImportError:
    print('GTLS is not installed! The search_GTLS class will not be available.')
    GTLS_active = False

active_searches = {'BLS': BLS_active, 'TLS': TLS_active, 'GTLS': GTLS_active}

if TLS_active:
    class search_TLS(transitleastsquares):
        def __init__(self, t, y, dy=None, verbose=True):
            super().__init__(t, y, dy, verbose)

        def search(self):
            1

if GTLS_active:
    class search_GTLS(transitleastsquares):
        def __init__(self, t, y, dy=None, verbose=True):
            super().__init__(t, y, dy, verbose)

        def search(self):
            1
try:
    from astropy.timeseries import BoxLeastSquares
    BLS_active = True
except ImportError:
    print('BoxLeastSquares is not installed! The search_BLS class will not be available.')
    BLS_active = False

try:
    from transitleastsquares import transitleastsquares
    TLS_active = True
except ImportError:
    print('TransitLeastSquares is not installed! The search_TLS class will not be available.')
    TLS_active = False

try:
    from gputls import gtls
    GTLS_active = True
except ImportError:
    print('GTLS is not installed! The search_GTLS class will not be available.')
    GTLS_active = False

active_searches = {'BLS': BLS_active, 'TLS': TLS_active, 'GTLS': GTLS_active}

class search_results:
    def __init__(self, ):
        1

if TLS_active:
    class search_TLS(transitleastsquares):
        def __init__(self, t, y, dy=None, verbose=True):
            super().__init__(t, y, dy, verbose)

        def search(self, **kwargs):
            self.results = 1

if GTLS_active:
    class search_GTLS(gtls):
        def __init__(self, t, y, dy=None, verbose=True):
            super().__init__(t, y, dy, verbose)

        def search(self):
            1

        def search_chunks(self, chunk_size=10000):
            1
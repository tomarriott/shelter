try:
    import juliet
    juliet_active = True
except ImportError:
    print('Juliet is not installed! The fit_juliet class will not be available.')
    juliet_active = False

class Prior:
    def __init__(name, ):
        1

class Priors:
    def __init__(self):
        self.priors = []

    def add_prior(self, *args):
        self.priors.append()
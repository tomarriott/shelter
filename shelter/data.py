import numpy as np
from dataclasses import dataclass, field

# Merge RV and LC classes
# TODO: Make these dataclasses?
class DataContainer:
    def __init__(self, t, y, e=[], instrument=None):

        if len(t) != len(y):
            raise ValueError('t and y must be same size')

        self.t = np.asanyarray(t)
        self.y = np.asanyarray(y)
        if e == []:
            e = np.zeros(len(y))
        self.e = e

        self.N = len(t)
        self.instrument = instrument

@dataclass
class DataContainer:
        t: np.ndarray = field(default=np.array([]))
        y: np.ndarray = field(default=np.array([]))
        e: np.ndarray = field(default=np.array([]))

        if e == np.array([]):
            e = np.zeros(len(y))

        N: int = len(t)
        instrument: str = field(default='')

        if len(t) != len(y):
            raise ValueError('t and y must be same size')

class LightCurve(DataContainer):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return f"LightCurve(N_points={self.N}, Instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"

    def to_lightkurve(self, **kwargs):
        return to_lightkurve(self, **kwargs)

class RadialVelocity(DataContainer):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return f"RadialVelocity(N_points={self.N}, Instrument={self.instrument}, t={self.t}, y={self.y}, e='{self.e})"
    
def to_lightkurve(lc, **kwargs):
    try:
        from lightkurve import LightCurve as lk_LightCurve
    except ImportError:
        print('Lightkurve is not installed! Skipping construction')
        return None

    return lk_LightCurve(time=lc.t, flux=lc.y, flux_err=lc.e, **kwargs)

def from_lightkurve(lc):
    return 

lc = LightCurve()
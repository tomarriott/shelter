import numpy as np
import traceback
from copy import deepcopy
from sklearn.cluster import MiniBatchKMeans, MeanShift, estimate_bandwidth
from PIL import Image
import os

def sRGB_to_RGB(rgb):
    """
    Convert a set of sRGB colour values to Linear RGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        RGB (array-like): Linear RGB value
    """

    return np.where(rgb>0.04045, ((rgb+0.055)/1.055)**2.4, rgb/12.92)

def RGB_to_sRGB(RGB):
    """
    Convert a set of Linear RGB colour values to sRGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        RGB (array-like): Linear RGB value

    Returns:
        rgb (array-like): sRGB value
    """

    return np.where(RGB>0.0031308, ((RGB**(1/2.4))*1.055)-0.055, RGB*12.92)

def HSV_to_sRGB(HSV):
    """
    Convert a set of HSV colour values to sRGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        HSV (array-like): HSV value

    Returns:
        rgb (array-like): sRGB value
    """

    def HSV_fn(HSV, n):
        k = np.mod(n + (HSV[0]*6), 6)
        return HSV[2] - (HSV[2] * HSV[1] * max(0, min(k, 4-k, 1)))
    
    return np.array([HSV_fn(HSV, 5), HSV_fn(HSV, 3), HSV_fn(HSV, 1)])

def sRGB_to_HSV(rgb):
    """
    Convert a set of sRGB colour values to HSV.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        HSV (array-like): HSV value
    """
    return 1
    #X_max = np.max()

def sRGB_to_Hex(rgb):
    """
    Convert a set of sRGB colour values to a Hex code.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        Hex (string or list of strings): Hex code
    """
    rgb = np.asarray(rgb)
    if len(rgb.shape) >= 2: 
        hexes = []
        for n in range(rgb.shape[1]):
            hex = '#' + ''.join('%02X' % round(i*255) for i in rgb[:, n])
            hexes.append(hex)
        return hexes
    return '#' + ''.join('%02X' % round(i*255) for i in rgb)

def XYZ_to_RGB(XYZ):
    """
    Convert a set of CIE XYZ colour values to Linear RGB.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value 

    Returns:
        RGB (array-like): Linear RGB value
    """
        
    transfer_matrix = np.array([[+3.2406255, -1.5372080, -0.4986286],
                                [-0.9689307, +1.8757561, +0.0415175],
                                [+0.0557101, -0.2040211, +1.0569959]])
    return np.matmul(transfer_matrix, np.array(XYZ))

def XYZ_to_HSV(XYZ):
    """
    Convert a set of CIE XYZ colour values to HSV.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value 

    Returns:
        HSV (array-like): HSV value
    """



def XYZ_to_Oklab(XYZ):
    """
    Convert a set of CIE XYZ colour values to Oklab.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        XYZ (array-like): CIE XYZ value

    Returns:
        Lab (array-like): Oklab value
    """

    transfer_matrix_1 = np.array([[+0.8189330101, +0.3618667424, -0.1288597137],
                                  [+0.0329845436, +0.9293118715, +0.0361456387],
                                  [+0.0482003018, +0.2643662691, +0.6338517070]])
    
    transfer_matrix_2 = np.array([[+0.2104542553, +0.7936177850, -0.0040720468],
                                  [+1.9779984951, -2.4285922050, +0.4505937099],
                                  [+0.0259040371, +0.7827717662, -0.8086757660]])
    
    return np.matmul(transfer_matrix_2, (np.matmul(transfer_matrix_1, np.array(XYZ))**(1/3)))

def sRGB_to_XYZ(rgb):
    """
    Convert a set of sRGB colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        rgb (array-like): sRGB value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    return RGB_to_XYZ(sRGB_to_RGB(rgb))

def RGB_to_XYZ(RGB):
    """
    Convert a set of Linear RGB colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        RGB (array-like): Linear RGB value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    transfer_matrix = np.array([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]])
    return np.matmul(transfer_matrix, np.array(RGB))

def HSV_to_XYZ(HSV):
    """
    Convert a set of HSV colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        HSV (array-like): HSV value

    Returns:
        XYZ (array-like): CIE XYZ value
    """
        
    return RGB_to_XYZ(sRGB_to_RGB(HSV_to_sRGB(HSV)))

def Oklab_to_XYZ(Lab):
    """
    Convert a set of Oklab colour values to CIE XYZ.
    Values must be defined between 0 and 1.
    If inputting arrays of values, the input arrays must be of the same length.

    Parameters:
        Lab (array-like): Oklab value

    Returns:
        XYZ (array-like): CIE XYZ value
    """

    transfer_matrix_1 = np.linalg.inv(np.array([[+0.8189330101, +0.3618667424, -0.1288597137],
                                                [+0.0329845436, +0.9293118715, +0.0361456387],
                                                [+0.0482003018, +0.2643662691, +0.6338517070]]))
    
    transfer_matrix_2 = np.linalg.inv(np.array([[+0.2104542553, +0.7936177850, -0.0040720468],
                                                [+1.9779984951, -2.4285922050, +0.4505937099],
                                                [+0.0259040371, +0.7827717662, -0.8086757660]]))
    
    return np.matmul(transfer_matrix_1, (np.matmul(transfer_matrix_2, np.array(Lab))**3))

# Append any new spaces implemented here please!
_to_XYZ_methods = {
    "sRGB": lambda v: sRGB_to_XYZ(v),
    "RGB": lambda v: RGB_to_XYZ(v),
    "HSV": lambda v: HSV_to_XYZ(v),
    "XYZ": lambda v: np.array(v),
    "Oklab": lambda v: Oklab_to_XYZ(v), 
}

_from_XYZ_methods = {
    "sRGB": lambda v: RGB_to_sRGB(XYZ_to_RGB(v)),
    "RGB": lambda v: XYZ_to_RGB(v),
    "HSV": lambda v: HSV_to_XYZ(v),
    "Hex": lambda v: sRGB_to_Hex(RGB_to_sRGB(XYZ_to_RGB(v))),
    "XYZ": lambda v: np.array(v),
    "Oklab": lambda v: XYZ_to_Oklab(v), 
}

def convert_colour(value, input_space, output_space):
    if input_space == output_space:
        return value
    return _from_XYZ_methods[output_space](_to_XYZ_methods[input_space](value))

class Colour:
    """A class for storing and manipulating colours."""

    def __init__(self, value, space, alpha=None):
        """
        Parameters:
            value (array-like of floats): 3 floats (e.g., (r, g, b) or (h, s, v))
            space (string): One of "sRGB", "RGB", "HSV", etc.
            alpha (float): Value for the alpha (opacity) channel. Defaults to 1.
        """
        value = np.asarray(value)

        if space not in _to_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')
            
        if alpha == None:
            if len(np.array(value).shape) == 1:
                alpha = 1
            else:
                alpha = np.ones(np.array(value).shape[1])

        self._value = _to_XYZ_methods[space](value)     # Colour value stored as an array
        self._space = "XYZ"     # All colours are internally stored in CIE XYZ space. Why is this a variable?
        self.alpha = alpha      # Alpha channel is stored separately
    
    def set_colour(self, value, space):
        """
        Set the colour in the object to a new colour.

        value (array-like): 3 floats (e.g., (r, g, b) or (h, s, v))
        space (string): One of "sRGB", "RGB", "HSV", etc.
        """
        value = np.asarray(value)

        if space not in _to_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')
            
        if alpha == None:
            alpha = np.ones(len(np.asarray([value[0]])))
        
        self._value = _to_XYZ_methods[space](value)     # Colour value stored as an array

    def set_alpha(self, alpha):
        """Set alpha value of colour. Here for consistency with set_colour."""
        self.alpha = alpha

    def get_colour(self, space, return_alpha=False):
        """
        Get the colour value in the desired colour space.

        space (string): Colour space to output into.
        return_alpha (bool, default False): If True, returns (colour, alpha).
        """
        if space not in _from_XYZ_methods:
            raise KeyError(f'Unsupported or invalid colour space: {space}')

        out = _from_XYZ_methods[space](self._value)
        if return_alpha:
            return out, self.alpha
        return out
    
    def get_alpha(self):
        """Get alpha value of colour. Here for consistency with get_colour."""
        return self.alpha


class Colour_Stop(Colour):
    """A class for the colour stops of a gradient."""

    def __init__(self, position, value, space, alpha=None):
        super().__init__(value, space, alpha)
        self.position = position


class Gradient:
    """A class for creating gradients through a desired colour space."""

    def __init__(self, colours, positions=None, interp_space='Oklab', cyclic_direction='near'):
        """
        Parameters:
            colours (array-like of Colours): Set of Colour objects to create the gradient with.
            positions (array-like of floats): Positions of the colour stops. Values must be defined between 0 and 1. By default colours will be spaced out equally.
            interp_space (string): Colour space to interpolate the gradient within. Default is 'Oklab'.
            cyclic_direction (string): How the gradient handles cyclic coordinates, e.g. hue in HSV. 
        """
        n = len(colours)
        if n <= 1:
            raise ValueError("Gradient requires at least 2 colours")    # Could maybe use 1 but that would be pointless
        
        if positions is None:
            positions = np.linspace(0.0, 1.0, n)
        else:
            positions = np.asarray(positions, dtype=float)
            if positions.shape[0] != n:
                raise ValueError(f"Number of positions ({positions.shape[0]}) does not match number of colours ({n})")
            if np.any(positions < 0.0) or np.any(positions > 1.0):
                raise ValueError("Positions must be between 0 and 1")

        # Check if chosen space works then add
        if any([interp_space not in _to_XYZ_methods, interp_space not in _from_XYZ_methods]):
            raise KeyError(f"Unsupported or invalid interpolation space: {interp_space}")

        # Generate the interpolation space colours (probably faster than re-computing each time a gradient is made)
        # TODO: although with this approach the user will have to call a method to change the interp space, probably. I'll add that later.
        stop_values = np.asarray([colour.get_colour(interp_space) for colour in colours])
        stop_alphas = np.asarray([colour.alpha for colour in colours]).reshape(-1, 1)
        
        # Sort the colours by position
        sort_order = np.argsort(positions)
        self.positions = np.asarray(positions)[sort_order]
        self.colours = np.asarray(colours)[sort_order]
        self._stop_values = np.hstack((stop_values, stop_alphas))[sort_order]
        self._interp_space = interp_space
        self._nstops = n
        self._most_recent = n - 1

    def _sample_gradient(self, t):
        """
        Sample the gradient at a point (or set of points).
        If the gradient is empty, picked colours will be solid black.
        """
        if self._nstops <= 1:
            if self._nstops == 0:
                return np.zeros(4, t)
            if self._nstops == 1:
                return np.ones(4, t) * self._stop_values[0]

        i = self.positions.searchsorted(t)
        print(self.positions, t, i)
        # Note: this will break if there is no stop above or below t
        # TODO: add checks for if this occurs
        frac = (t - self.positions[i-1]) / (self.positions[i] - self.positions[i-1])
        values = self._stop_values[i-1] + (frac * (self._stop_values[i] - self._stop_values[i-1]))
        print(values)
        return values

    def add_stop(self, colour=None, position=None):
        """
        Add a new colour stop to the gradient.
        If colour is None then new colour will be interpolated from the existing gradient at the chosen position.
        If position is None then the stop will be added in the centre of the largest gap, or at either end if 1 or fewer stops are present.

        Parameters:
            colour (Colour): Colour object to assign to the stop.
            position (float): Position of the colour stop. Value must be defined between 0 and 1.
        """
        if position == None:
            if self._nstops <= 1:
                if any([self._nstops == 1 and self.positions[0] > 0.5, self._nstops == 0]):
                    position = 0.0
                elif self._nstops == 1 and self.positions[0] <= 0.5:
                    position = 1.0
            else:
                gaps = np.diff(self.positions)
                position = self.positions[gaps.argmax()] + (gaps.max()/2)
        
        if colour == None:
            if self._nstops == 0:
                colour = Colour([0, 0, 0], "RGB")
            if self._nstops == 1:
                colour = deepcopy(self.colours[0])
            else:
                colour = self._sample_gradient(position)

        stop_value = np.append(colour.get_colour(self._interp_space), colour.alpha)

        idx = self.positions.searchsorted(position)
        self.positions = np.concatenate((self.positions[:idx], [position], self.positions[idx:]))
        self.colours = np.concatenate((self.colours[:idx], [colour], self.colours[idx:]))
        self._stop_values = np.concatenate((self._stop_values[:idx], [stop_value], self._stop_values[idx:]))
        self._nstops += 1
        self._most_recent = idx

    def remove_stop(self, index=None, position=None):
        """
        Remove a colour stop from the gradient.
        Choose to remove it either by index (preferred) or by closest position.
        If neither is defined, the most recent stop added to the gradient will be removed.

        Parameters:
            index (integer): Index of the colour stop to remove, ordered by position.
            position (float): Approximate position of the colour stop to remove. Value must be defined between 0 and 1.
        """
        if index == None:
            if position == None:
                index = self._most_recent
            else:
                index = (np.abs(self.positions - position)).argmin()

        self.positions = np.delete(self.positions, index)
        self.colours = np.delete(self.colours, index)
        self._stop_values = np.delete(self._stop_values, index)
        self._nstops -= 1

    def move_stop(self, new_position, index=None, position=None):
        """
        Move a colour stop in the gradient to a new position.
        Choose to move it either by index (preferred) or by closest position.
        If neither is defined, the most recent stop added to the gradient will be moved.

        Parameters:
            new_position (float): 
            index (integer): Index of the colour stop to remove, ordered by position.
            position (float): Approximate position of the colour stop to remove. Value must be defined between 0 and 1.
        """

    def edit_stop(self, new_colour, index=None, position=None):
        """
        Edit a colour stop in the gradient.
        Choose to edit it either by index (preferred) or by closest position.
        If neither is defined, the most recent stop added to the gradient will be edited.

        Parameters:
            new_colour (Colour): 
            index (integer): Index of the colour stop to remove, ordered by position.
            position (float): Approximate position of the colour stop to remove. Value must be defined between 0 and 1.
        """

colour = Colour([0.1, 0.1, 0.1], "RGB")
print(colour.get_colour("Oklab"))
print(colour.get_colour("Hex"))
print(colour.get_alpha())
        
colour = Colour([np.linspace(0.1, 0.1, 8), np.linspace(0.1, 0.1, 8), np.linspace(0.1, 0.1, 8)], "RGB")
print(colour.get_colour("Oklab"))
print(colour.get_colour("Hex"))
print(colour.get_alpha())

print("Gradients:")
colour1 = Colour([0, 0.1, 0.2], "RGB")
colour2 = Colour([1, 0.1, 0.2], "RGB")
gradient = Gradient([colour1, colour2])

print("Adding a colour:")
#gradient.add_stop()
#print(gradient.positions)
#print(gradient.colours)

print("Removing a colour:")
#gradient.remove_stop()
#print(gradient.positions)
#print(gradient.colours)
#print([colour.get_colour(gradient._interp_space) for colour in gradient.colours])

def coloured_square(hex_string):
    """
    Returns a coloured square that you can print to a terminal.
    """
    hex_string = hex_string.strip("#")
    assert len(hex_string) == 6
    red = int(hex_string[:2], 16)
    green = int(hex_string[2:4], 16)
    blue = int(hex_string[4:6], 16)

    return f"\033[48:2::{red}:{green}:{blue}m \033[49m"

def k_means_clustering(data, n_clusters=8, n_runs=1, seed=None, **kwargs):
    """
    Perform K-Means Clustering on a set of colour values, e.g. an image.
    
    Parameters:
        data (array): 2D array of colour values to cluster.
        n_clusters (int): Number of clusters to create.
        n_runs (int): Number of runs of the algorithm to perform. The run with the lowest overall distance will be returned.
        seed (int, or array-like of int): Random seed for the clustering. If None, will be random (resulting in a non-deterministic outcome).
        Must be an array of length equal to n_runs if n_runs > 1.
    """
    distances = []
    clusters = []
    labels = []
    for i in range(n_runs):
        if seed is not None:
            seed = np.asarray(seed)[i]
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, **kwargs).fit(data)
        distances.append(kmeans.inertia_)
        clusters.append(kmeans.cluster_centers_)
        labels.append(kmeans.labels_)

    best_fit = np.argmax(np.asarray(distances))
    return clusters[best_fit], labels[best_fit]

def mean_shift_clustering(data, quantile=0.1, n_samples=5000, bin_seeding=True, min_bin_freq=5, n_jobs=4, **kwargs):
    """
    Perform Mean-Shift Clustering on a set of colour values, e.g. an image.
    
    Parameters:
        data (array): 2D array of colour values to cluster.
    """
    #bandwidth = 0.08
    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)
    clustering = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding, min_bin_freq=min_bin_freq, n_jobs=n_jobs, **kwargs).fit(data)

    return clustering.cluster_centers_, clustering.labels_

def cluster_image(img, clustering_space='Oklab', clustering_method='mean-shift', input_space='sRGB', output_space='sRGB', return_labels=False, **kwargs):
    img_flat = np.asarray([convert_colour(value, input_space, clustering_space) for value in img.reshape(img.shape[0] * img.shape[1], 3)/255])

    if clustering_method == 'k-means':
        clusters, labels = k_means_clustering(img_flat, **kwargs)
    if clustering_method == 'mean-shift':
        clusters, labels = mean_shift_clustering(img_flat, **kwargs)

    img_clusters = np.asarray([convert_colour(value, clustering_space, output_space) for value in clusters])

    if return_labels:
        return img_clusters, labels
    return img_clusters
'''
img = np.asarray(Image.open("G:\\My Drive\\Obsidian\\Notes\\50 - Media\\51 - Albums\\Cover Art\\Hellfire.jpg"))
img = img[::6, ::6]

print('K-Means Clustering:')
img_clusters = cluster_image(img, 'Oklab', clustering_method='k-means', n_runs=10)
img_cluster_colours = [convert_colour(value, 'sRGB', 'Hex') for value in img_clusters]
img_cluster_blocks = [coloured_square(hex) for hex in img_cluster_colours]
for i in range(len(img_cluster_blocks)):
    print(img_cluster_blocks[i], img_cluster_colours[i])


print('Mean-Shift Clustering:')
img_clusters = cluster_image(img, 'Oklab', clustering_method='mean-shift')
img_cluster_colours = [convert_colour(value, 'sRGB', 'Hex') for value in img_clusters]
img_cluster_blocks = [coloured_square(hex) for hex in img_cluster_colours]
for i in range(len(img_cluster_blocks)):
    print(img_cluster_blocks[i], img_cluster_colours[i])
'''
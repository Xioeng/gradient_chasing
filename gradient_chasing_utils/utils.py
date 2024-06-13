import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
import matplotlib.pyplot as plt
import pickle
import sys
from scipy.interpolate import griddata
from shapely.geometry import Point, Polygon


class Normalizer:
    """
    Normalizes and denormalizes coordinates.

    Args:
        mins_maxs (tuple): Tuple containing the minimum values of the coordinates.
        maxs (tuple): Tuple containing the maximum values of the coordinates.
        dom_maxs (numpy.ndarray): Maximum values of the domain.

    Attributes:
        mins (numpy.ndarray): Minimum values of the coordinates.
        maxs (numpy.ndarray): Maximum values of the coordinates.
        dom_maxs (numpy.ndarray): Maximum values of the domain.
    """
    def __init__(self, mins =(25.7581572, -80.3734494), maxs = (25.7583659, -80.3738642), dom_maxs=np.array([1.0, 1.0])):
        self.mins, self.maxs = np.asarray(mins), np.asarray(maxs)
        self.dom_maxs = dom_maxs

    def forward(self, x):
        """
        Forward normalization. (From coordinates domain to transformed domain)

        Args:
        - x: Input coordinates.

        Returns:
        - Normalized coordinates.
        """
        x = np.asarray(x)
        return self.dom_maxs * (x - self.mins) / (self.maxs - self.mins) 
    
    def inverse(self, x):
        """
        Inverse normalization.

        Args:
        - x: Normalized coordinates.

        Returns:
        - Inverse normalized coordinates.
        """
        x = np.asarray(x)
        return (x / self.dom_maxs) * (self.maxs - self.mins) + self.mins
    

def sort_coordinates(coords):
    """
    Sorts coordinates in counterclockwise sense.

    Args:
        coords (numpy.ndarray): Coordinates to be sorted.

    Returns:
        numpy.ndarray: Sorted coordinates.
    """
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:,1] - centroid[1], coords[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords

def mask_from_polygon(meshgrid, polygon_coordinates):
    """
    Create a mask matrix indicating whether points in a meshgrid are inside a polygon.
    Args:
    - meshgrid: Tuple of arrays from np.meshgrid.
    -polygon_coordinates: Polygon's coordinates; sorted counterclockwise.

    Returns:
    - numpy.ndarray: Mask from meshgrid
    """
    X, Y = meshgrid
    mask = np.zeros_like(X, dtype=bool)
    polygon = Polygon(polygon_coordinates)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = Point(X[i, j], Y[i, j])
            mask[i, j] = polygon.contains(point)
    
    return mask



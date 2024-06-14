import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct
from numpy.linalg import norm
from utils import Normalizer
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import pickle

class WaterPhenomenonGP:
    """
    WaterPhenomenonGP models water-related phenomena using Gaussian Processes.

    This class uses a Gaussian Process (GP) to model and make predictions about water-related phenomena
    based on latitude and longitude coordinates. It normalizes the input data, fits the GP model, and 
    computes gradients to suggest the next point for analysis.

    Attributes:
        _normalizer (Normalizer): An instance of the Normalizer class to handle data normalization.
        _gaussian_process (GaussianProcessRegressor): The Gaussian Process model for regression.
    """
    
    def __init__(self, mins, maxs, kernel = RBF(), workspace_latitude = 25.7617):
        """
        Initializes the WaterPhenomenonGP with normalization parameters and a kernel for the GP.

        Args:
            mins (tuple, numpy.array): Minimum coordinates of the domain of interest. (latitude, longitude)
            maxs (tuple, numpy.array): Maximum coordinates of the domain of interest.
            kernel (sklearn.gaussian_process.kernels): The kernel to be used by the Gaussian Process.
        """
        # Extract latitude and longitude boundaries for normalization
        dim_meters = (np.asarray(maxs) - np.asarray(mins)) * 111111 * np.array([1.0, np.cos(np.radians(workspace_latitude))])
        # print(dim_meters)
        
        # Initialize the normalizer
        self._normalizer = Normalizer(mins, maxs, dim_meters)
        
        # Initialize the Gaussian Process regressor
        self._gaussian_process = GaussianProcessRegressor(kernel=kernel,
                                                          n_restarts_optimizer=5,
                                                          alpha=1e-5)
    
    def _tap_gradient(self, x, h=0.001):
        """
        Computes the gradient of the GP predictions at point x using finite differences.

        Args:
            x (numpy.ndarray): The input point where the gradient is to be computed.
            h (float): The perturbation step size for finite differences (default is 0.001).

        Returns:
            numpy.ndarray: The gradient vector at point x.
        """
        x = x.reshape(1, -1)
        dim = x.shape[-1]
        perturbations = h * np.eye(dim)
        
        # Compute predictions for perturbed points
        predictions_plus = self._gaussian_process.predict(x + perturbations)
        predictions_minus = self._gaussian_process.predict(x - perturbations)
        
        # Calculate gradient using central differences
        grad = (predictions_plus - predictions_minus) / (2 * h)
        
        return grad.flatten()

    def fit(self, X, y):
        """
        Fits the Gaussian Process model to the training data.

        Args:
            X (numpy.ndarray): The input training data (coordinates).
            y (numpy.ndarray): The target values corresponding to X.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Normalize the input data
        X_transform = self._normalizer.forward(X)
        
        # Fit the GP model
        self._gaussian_process.fit(X_transform, y)

    def next_point(self, x, lr=1.0):
        """
        Suggests the next point for analysis based on the gradient of the GP predictions.

        Args:
            x (numpy.ndarray): The current point (coordinates).
            lr (float): The learning rate determining the step size in the direction of the gradient (default is 1.0).

        Returns:
            numpy.ndarray: The suggested next point (coordinates).
        """
        x_inv = self._normalizer.forward(x)
        gradient = self._tap_gradient(x_inv)

        # Determine the direction based on the gradient
        if norm(gradient) > 1e-4:
            direction = gradient / norm(gradient)
        else:
            print('No movement')
            direction = np.array([0., 0.])
        
        # Compute the next point in the normalized space and transform it back
        next_point_inv = x_inv + lr * direction
        return self._normalizer.inverse(next_point_inv)


def plot_env_and_path(environment, path, extent, delta=0.0004,
                    plot_args={'marker': 'x', 'color': 'black'},
                    contourf_args={'cmap': 'jet', 'alpha': 0.5, 'levels' : 15}):
    """
    Plots the interpolated environment data and the path on a satellite image.

    Parameters:
    - environment: Callable
        Function to evaluate the environment at given coordinates (latitude, longitude).
    - path: np.ndarray
        Array of coordinates representing the path (shape: [n_points, 2]).
    - extent: tuple
        Tuple specifying the extent of the plot (min_lat, min_lon, max_lat, max_lon).
    - delta: float, optional
        Margin to add around the extent for better visualization (default is 0.00015).
    - plot_args: dict, optional
        Additional arguments for the path plot (default is empty dict).
    - contourf_args: dict, optional
        Additional arguments for the contour plot (default is empty dict).

    Returns:
    - None
    """
    min_lat, min_lon, max_lat, max_lon = extent

    # Create meshgrid for the specified extent
    lat_grid, lon_grid = np.meshgrid(np.linspace(min_lat, max_lat, 100), np.linspace(min_lon, max_lon, 100))
    mesh_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))

    # Evaluate the environment function on the meshgrid points
    interpolated_values = environment(mesh_points).reshape(lat_grid.shape)

    # Initialize the map with satellite imagery if not already initialized
    if plot_env_and_path.fig is None or plot_env_and_path.ax is None:
        tiler = cimgt.GoogleTiles(style='satellite')
        transform = ccrs.PlateCarree()
        plot_env_and_path.fig, plot_env_and_path.ax = plt.subplots(figsize=(14, 7), subplot_kw={'projection': transform})
        plot_env_and_path.ax.add_image(tiler, 21)
        plot_env_and_path.ax.set_aspect('equal', adjustable='box')
        gl = plot_env_and_path.ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

    # Set the extent of the map with a margin
    plot_env_and_path.ax.set_extent([min_lon - delta, max_lon + delta, min_lat - delta, max_lat + delta], crs=ccrs.PlateCarree())

    # Plot the path
    plot_env_and_path.ax.plot(path[:, 1], path[:, 0], transform=ccrs.PlateCarree(), **plot_args)

    # Plot the interpolated environment values as contours
    try:
        for c in plot_env_and_path.c.collections:
            c.remove()
    except:
        pass
    plot_env_and_path.c = plot_env_and_path.ax.contourf(lon_grid, lat_grid, interpolated_values, transform=ccrs.PlateCarree(), **contourf_args)
    plt.draw()
    plt.pause(0.1)

# Initialize static variables for the function
plot_env_and_path.fig = None
plot_env_and_path.ax = None
plot_env_and_path.c = None


def load_solution_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    # Example
    kernel = RBF()
    X = np.array([
    [25.7617, -80.1918],  # Miami Downtown
    [25.7618, -80.1917],  # 10 meters north-east
    [25.7616, -80.1919],  # 10 meters south-west
    [25.7617, -80.1919],  # 10 meters west
    [25.7618, -80.1918]   # 10 meters north
    ], dtype = np.float64)

    # Example temperature values (randomly generated)
    y = np.random.uniform(25, 30, size=5)

    water_feature = WaterPhenomenonGP(np.min(X, axis = 0), np.max(X, axis = 0), kernel)

    water_feature.fit(X,y)

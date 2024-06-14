import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF
import pickle
from utils import Normalizer, sort_coordinates
from utils_gp_gradient import WaterPhenomenonGP, plot_env_and_path
import matplotlib.pyplot as plt

def load_solution_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)[0][-1]

def initialize_gp(polygon_coords, kernel):
    sorted_coords = sort_coordinates(polygon_coords)
    min_coords = np.min(sorted_coords, axis=0)
    max_coords = np.max(sorted_coords, axis=0)
    water_feature = WaterPhenomenonGP(min_coords, max_coords, kernel)
    return water_feature, min_coords, max_coords

def define_environment(solution, normalizer):
    return lambda x: solution(normalizer.forward(x))

def fit_and_predict(water_feature, X, y, min_coords, max_coords, iterations=20, environment = None, plot_env_mode = 'same'):
    for i in range(iterations):
        water_feature.fit(X, y)
        next_point = water_feature.next_point(X[-1], 3.0) + 0.00001*np.random.rand(2,)
        X = np.vstack([X, next_point])
        print(environment(next_point))
        y = np.vstack([y, environment(next_point)])
        print(next_point, y[-1])
        if plot_env_mode == 'same':
            plot_env = environment
        else:
            plot_env = lambda x: water_feature._gaussian_process.predict(water_feature._normalizer.forward(x))

        plot_env_and_path(plot_env, X, (*min_coords, *max_coords), 0.0002)
        plt.pause(0.5)


# Main script
polygon_coordinates = np.array([[25.7581072, -80.3738942],
                                [25.7581072, -80.3734494],
                                [25.7583659, -80.3738942],
                                [25.7583659, -80.3734494]])

# Load the temperature solution
solution = load_solution_from_pkl('temperature_test.pkl')

# Define the kernel
kernel = 10 * Matern(nu=0.5, length_scale_bounds=(1e-2, 1e5)) + 1e-2 * DotProduct() ** 1
kernel = Matern()

# Initialize the Gaussian Process
water_feature, min_coords, max_coords = initialize_gp(polygon_coordinates, kernel)

# Create a normalizer
normalizer = water_feature._normalizer

# Define the environment function
environment = define_environment(solution, normalizer)

A=10; sigma_x=0.0001; sigma_y=0.0001; x0, y0 = [25.7582, -80.37360]
def environment(x): 
    x = np.atleast_2d(x).T
    return A * np.exp(-((x[0] - x0)**2 / (2 * sigma_x**2)) - ((x[1] - y0)**2 / (2 * sigma_y**2)))


# Initial points for fitting
initial_points = normalizer.inverse(np.array([[1, 1], [5, 1], [1, 5], [5, 5]]) + np.array([10, 5]))

initial_values = np.array([environment(x) for x in initial_points])

# Fit the model and predict the next points
if __name__ == '__main__':
    fit_and_predict(water_feature, initial_points, initial_values, min_coords, max_coords, iterations= 40, environment = environment, plot_env_mode = 'gp')
    plt.show()
    plt.pause(0.25)

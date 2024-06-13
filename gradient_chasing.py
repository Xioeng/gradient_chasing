import surveyor_library.surveyor_helper as hlp
from surveyor_library import Surveyor
from gradient_chasing_utils import utils_gp_gradient

import sys
import time
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.gaussian_process.kernels import DotProduct, Matern
import matplotlib.pyplot as plt


def allocate_data_df(boat):
    """
    Allocate a DataFrame to store the boat's data.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.

    Returns:
        pd.DataFrame: A DataFrame containing the boat's initial data.
    """
    return pd.DataFrame([boat.get_data()])

def start_mission(boat):
    """
    Start the mission by waiting for the operator to switch to waypoint mode.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.
    """
    print('Ready to start the mission! Switch manually to waypoint mode')
    boat.set_standby_mode()

    while boat.get_control_mode() != "Waypoint":
        pass

    countdown(5, "Starting mission in", "Change the operator to secondary mode!")
    print('Mission started!')

def countdown(count, message, additional_message=""):
    """
    Print a countdown with the given message and optional additional message.

    Args:
        count (int): The number of seconds to count down.
        message (str): The message to display before the countdown.
        additional_message (str, optional): An additional message to display after the countdown.
    """
    for i in range(count, 0, -1):
        print(f'{message} {i}. {additional_message}', end="\r")
        time.sleep(1)
    print()   

def load_and_send_waypoint(boat, waypoint, erp, throttle):
    """
    Load the next waypoint and send it to the boat.

    Args:
        boat (Surveyor): The Surveyor object representing the boat.
        waypoint (tuple): The waypoint coordinates to be sent.
        erp (list): A list of ERP coordinates.
        throttle (int): The desired throttle value for the boat.

    """
    boat.send_waypoints([waypoint], erp, throttle)

    while boat.get_control_mode() != 'Waypoint':
        dist = geodesic(waypoint, boat.get_gps_coordinates()).meters
        print(f'Distance to next waypoint {dist}')
        boat.set_waypoint_mode()
        if dist <= 2.0:
            break
        

FEATURE_TO_CHASE = 'ODO (mg/L)'
def data_updater(boat, mission_postfix = ''):
    global DATA
    data_dict = boat.get_data()
    DATA = pd.concat([DATA,
                    pd.DataFrame([data_dict])])
    print(DATA[['Latitude', 'Longitude', FEATURE_TO_CHASE]])
    hlp.save(data_dict, mission_postfix)

def next_waypoint(step_size = 4.5 ):
    global DATA
    X = np.asarray(DATA[['Latitude', 'Longitude']])
    y = np.asarray(DATA[FEATURE_TO_CHASE])
    water_feature_gp.fit(X, y)
    utils_gp_gradient.plot_env_and_path(
        environment = lambda x: water_feature_gp._gaussian_process.predict(water_feature_gp._normalizer.forward(x)), #To be explained, Jose's stuff
        path = X,
        extent = (*MINS, *MAXS)
    )
    return water_feature_gp.next_point(X[-1], step_size)

plt.ion()

# GP initialization
kernel = 10 * Matern(nu=0.5, length_scale_bounds=(1e-2, 1e5)) + 1e-2 * DotProduct() ** 1
extent_coordinates = np.array([[25.7581072, -80.3738942],
                                [25.7581072, -80.3734494],
                                [25.7583659, -80.3738942],
                                [25.7583659, -80.3734494]])

MAXS, MINS = np.max(extent_coordinates, axis = 0), np.min(extent_coordinates, axis = 0)
water_feature_gp = utils_gp_gradient.WaterPhenomenonGP(MINS, MAXS, kernel)
THROTTLE = 30; DATA = pd.DataFrame()

def main(filename, erp_filename, mission_postfix= ""):
    print(f'Reading waypoints from {filename} and ERP from {erp_filename}')
    initial_waypoints = hlp.read_csv_into_tuples(filename)
    
    erp = hlp.read_csv_into_tuples(erp_filename)
    boat = Surveyor()
    
    print(f'{len(initial_waypoints)} initial waypoints')

    with boat:
        start_mission(boat)
        for initial_waypoint in initial_waypoints:
            boat.go_to_waypoint(initial_waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                print(f'Initial collection mission waypoint {initial_waypoint}', end="\r")

            data_updater(boat, mission_postfix = mission_postfix) # Finished, getting data

        print('Starting gradient chasing')
        for i in range(30):
            waypoint = next_waypoint(DATA)
            print(f'Loading waypoint {i + 1}')
            boat.go_to_waypoint(waypoint, erp, THROTTLE)

            while boat.get_control_mode() == 'Waypoint':
                print(f'Navigating to waypoint {i + 1}', end="\r")
                # When you break this while loop you should have reached the waypoint, ready to assign a new waypoint

            data_updater(boat, mission_postfix = mission_postfix)

    plt.ioff() 
    plt.show()
            
if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: gradient_chasing.py <filename> <erp_filename> <mission_postfix>")
        sys.exit(1)

    main(*sys.argv[1:])
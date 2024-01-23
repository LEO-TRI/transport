import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import MiniBatchKMeans
from scripts_thesis.utils import to_array
from math import radians, sin, cos, sqrt, atan2

class CoordinateCounterTree:
    def __init__(self, data_coordinates : list, ball_tree: BallTree):
        # Initialize the class with geographical coordinates
        self.data_coordinates = data_coordinates
        self.ball_tree = ball_tree
        self.centroids = None

    @classmethod
    def from_data_points(self, latitude: list, longitude: list) -> "CoordinateCounterTree":

        data_coordinates = np.concatenate([to_array(latitude), to_array(longitude)], axis=1)
        data_coordinates = np.radians(data_coordinates)

        #returns a list of tuples (lat, lon)
        #data_coordinates = list(map(tuple, data_coordinates))

        ball_tree = BallTree(data_coordinates, metric="haversine")

        return CoordinateCounterTree(data_coordinates, ball_tree)


    def calculate_points_within_distance(self, point_coordinates: np.ndarray, distance_km: float=1.0):

        distance_radians = distance_km / 6371.0

        count = self.ball_tree.query_radius(np.radians(point_coordinates), r = distance_radians, return_distance=False)

        return count

    def calculate_centroids(self, point_coordinates: np.ndarray):

        kmeans = MiniBatchKMeans(n_clusters=2,
                                  random_state=1830,
                                  n_init="auto").fit(point_coordinates)

        self.centroids = kmeans.cluster_centers_

        return kmeans

def vectorized_distance(df : pd.DataFrame, lat : str, lon: str):

    df[[lat, lon]] = np.radians(df[[lat, lon]])
    shifted_df = df[[lat, lon]].shift(1)

    distance_df = df[[lat, lon]] - shifted_df

    a = (np.sin(distance_df[lat]/2)**2 + np.cos(df[lat]) * np.cos(shifted_df[lat]) * np.sin(distance_df[lon]/2)**2)**0.5
    c = 2 * np.arcsinh(a)
    return c * 6371.0


def haversine_distance(coord_1 : list, coord_2 : list):
    """
    Calculate the Haversine distance between two points specified by latitude and longitude.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point (in degrees)
    - lat2, lon2: Latitude and longitude of the second point (in degrees)

    Returns:
    - Distance in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = map(radians, coord_1)
    lat2, lon2 = map(radians, coord_2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate distance
    distance = 6371.0 * c

    return distance

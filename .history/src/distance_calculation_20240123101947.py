import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import MiniBatchKMeans
from math import radians, sin, cos, sqrt, atan2

class CoordinateCounterTree:
    def __init__(self, data_coordinates : list, ball_tree: BallTree):

        self.data_coordinates = data_coordinates
        self.ball_tree = ball_tree
        self.centroids = None

    @classmethod
    def from_data_points(self, latitude: list | np.ndarray, longitude: list | np.ndarray) -> "CoordinateCounterTree":
        """
        Initiate the ball tree

        Parameters
        ----------
        latitude : list
            The latitude points with which to initiate the tree, need to be a 1D array or a list
        longitude : list
            The longitude points with which to initiate the tree, need to be a 1D array or a list

        Returns
        -------
        CoordinateCounterTree
            The instantiated tree
        """

        data_coordinates = np.concatenate([np.asarray(latitude), np.asarray(longitude)], axis=1)
        data_coordinates = np.radians(data_coordinates)

        #returns a list of tuples (lat, lon)
        #data_coordinates = list(map(tuple, data_coordinates))

        ball_tree = BallTree(data_coordinates, metric="haversine")

        return CoordinateCounterTree(data_coordinates, ball_tree)


    def calculate_points_within_distance(self, point_coordinates: np.ndarray, distance_km: float=1.0) -> np.ndarray:
        """
        Calculates points within a given radius

        Parameters
        ----------
        point_coordinates : np.ndarray
            The coordinates of the points with wich to calculate the distance
        distance_km : float, optional
            The radius in km in which to check, by default 1.0

        Returns
        -------
        np.ndarray
            An array with for each point the index values of points falling in the radius
        """

        distance_radians = distance_km / 6371.0

        count = self.ball_tree.query_radius(np.radians(point_coordinates), r = distance_radians, return_distance=False)

        return count

    def calculate_centroids(self, point_coordinates: np.ndarray):

        kmeans = MiniBatchKMeans(n_clusters=2,
                                  random_state=1830,
                                  n_init="auto").fit(point_coordinates)

        self.centroids = kmeans.cluster_centers_

        return kmeans


def haversine_distance(coord_1 : list, coord_2 : list):
    """
    Calculate the Haversine distance between two points specified by latitude and longitude.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point (in degrees)
    - lat2, lon2: Latitude and longitude of the second point (in degrees)

    Returns:
    - Distance in kilometers
    """

    lat1, lon1 = map(radians, coord_1)
    lat2, lon2 = map(radians, coord_2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = 6371.0 * c

    return distance

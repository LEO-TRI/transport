import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

def load_europe_map(xlim : list, ylim : list, step : int, ship_coordinates : list[tuple], ship_data: list | np.ndarray) -> "gpd.GeoDataFrame":
    """
    Build a map of Europe and divide it in equal sized rectangles
    
    Then check for each pair ship/time where the point falls in

    Parameters
    ----------
    xlim : list
        The limit for the longitude coordinates, must be size 2
    ylim : list
        The limit for the latitude coordinates, must be size 2
    step : int
        The step with which to create the rectangles. 
        Defines number of rectangles with (xlim[1] - xlim[0]) / step
    ship_coordinates : list[tuple]
        A list of size 2 tuples of format  (lon, lat)
    ship_data : list | np.ndarray
        A list of data to be used when attributing emissions to geographical
        locations. Should be a list or 1D array
        
    Returns
    -------
    gpd.GeoDataFrame
        A geo dataframe composed of N rectangles with associated number of ships and emissions
    """

    # Calculate the step size for longitude and latitude
    lon_step = (xlim[1] - xlim[0]) / step
    lat_step = (ylim[1] - ylim[0]) / step

    ship_geometry = [Point(xy) for xy in ship_coordinates]

    #geoms = [(point, emission) for point, emission in zip(ship_geometry, ship_data)]

    squares_list = []
    for i in range(step):

        for j in range(step):

            lon_left = xlim[0] + (i * lon_step)
            lat_bottom = ylim[0] + (j * lat_step)
            lon_right = lon_left + lon_step
            lat_top = lat_bottom + lat_step

            square = Polygon([(lon_left, lat_bottom),
                              (lon_right, lat_bottom),
                              (lon_right, lat_top),
                              (lon_left, lat_top)
                              ]
                             )
            squares_list.append(square)

    n_squares = len(squares_list)
    squares_count = [0] * n_squares
    square_emissions = [0] * n_squares

    ship_iterator = zip(ship_geometry, ship_data, strict=True)
    for k, (row, emission) in enumerate(ship_iterator):

        for i, square in enumerate(squares_list):

            if square.contains(row):
                squares_count[i] += 1
                square_emissions[i] += emission
                break

        if k % 10**4 == 0:
            print(f"Finished {k} rows")

    tmp_ = pd.DataFrame({"count":squares_count, "emissions":square_emissions, "geometry":squares_list})
    return gpd.GeoDataFrame(data = tmp_.loc[:,:"emissions"], geometry = tmp_["geometry"])

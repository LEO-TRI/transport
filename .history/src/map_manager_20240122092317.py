import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon, Point

def load_europe_map(xlim : list, ylim : list, step : int, ship_coordinates : tuple, ship_data: list):

    xlim = xlim  # longitude range
    ylim = ylim  # latitude range

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

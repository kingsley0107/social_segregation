# -*- coding: utf-8 -*-
"""
Created on 06 Apr 3:08 PM

@Author: kingsley leung
@Email: kingsleyl0107@gmail.com

_description_: load origin data
"""
import geopandas as gpd
from geofeather import from_geofeather
import pandas as pd
import transbigdata as tbd

BOUNDARY_PATH = r"./data/boundary/shenzhen.geojson"
HOUSE_PATH = r"./data/house/shenzhen_filter.geojson"
MOBILE_PATH = r"./data/phones/mobile_data_filtered.feather"
RANKING = r"./data/UUID_rank_new.csv"
test = True
TIMEWINDOW = False
if test:
    # DISTANCE_MATRIX_PATH = r"./data/similarity_matrix_subset2.csv"
    DISTANCE_MATRIX_PATH = r"./data/similarity_matrix_test_reverse.csv"
else:
    DISTANCE_MATRIX_PATH = r"./data/similarity_matrix.csv"
GRID_SIZE = 1000

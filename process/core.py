# -*- coding: utf-8 -*-
"""
Created on 06 Apr 3:24 PM

@Author: kingsley leung
@Email: kingsleyl0107@gmail.com

_description_:
"""
import geopandas as gpd
import pandas as pd
import transbigdata as tbd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time
import numpy as np
import bson


# gen individual rank from housing price
def generate_rank(accuracy, BOUNDARY, HOUSE, MOBILE):
    grids_for_rank, params_for_rank = tbd.area_to_grid(
        BOUNDARY, accuracy=accuracy)
    grids_for_rank["gridID"] = grids_for_rank.index
    MOBILE["lon"] = MOBILE["geometry"].apply(lambda x: x.x)
    MOBILE["lat"] = MOBILE["geometry"].apply(lambda x: x.y)
    stay, move = tbd.mobile_stay_move(
        MOBILE, params_for_rank, ["uuid", "start_time", "lon", "lat"]
    )
    home_detected = tbd.mobile_identify_home(
        stay, ["uuid", "stime", "etime", "LONCOL", "LATCOL"]
    )
    user_home = gpd.GeoDataFrame(
        home_detected.merge(grids_for_rank, on=["LONCOL", "LATCOL"])
    )
    user_home["geometry"] = user_home["geometry"].apply(lambda x: x.centroid)
    matched = (
        tbd.ckdnearest_point(user_home, HOUSE)[
            ["uuid", "LONCOL", "LATCOL", "价格"]]
        .rename(columns={"价格": "price"})
        .sort_values(["price"])
        .reset_index(drop=True)
    )
    matched["rank"] = matched.index + 1
    matched = matched[["uuid", "LONCOL", "LATCOL", "rank", "price"]]
    return matched

# read social similarity matrix from preprocessed file


def get_Social_similarity_matrix(DISTANCE_MATRIX_PATH):
    DISTANCE_MATRIX = pd.read_csv(DISTANCE_MATRIX_PATH, index_col=[0])
    return DISTANCE_MATRIX

# extracting home location


def mobile_to_movement(mobile_data, params):
    if "lon" not in mobile_data.columns.tolist():
        mobile_data["lon"] = mobile_data["geometry"].x
    if "lat" not in mobile_data.columns.tolist():
        mobile_data["lat"] = mobile_data["geometry"].y
    stay_df, move_df = tbd.mobile_stay_move(
        mobile_data, params, col=["uuid", "start_time", "lon", "lat"]
    )
    return stay_df

# calculating individual's location probability in each grid(spatial unit)


def get_prob_matrix(stay_df, grids, RANK):
    grids["gridID"] = grids["gridID"].astype("int64")
    stay_df_merged = stay_df.merge(
        grids[["LONCOL", "LATCOL", "gridID"]], on=["LONCOL", "LATCOL"]
    ).merge(RANK, left_on="uuid", right_on="uuid")[
        ["uuid", "gridID", "rank", "duration"]
    ]
    observated_matrix = (
        stay_df_merged.groupby(["rank", "gridID"])
        .count()[["duration"]]
        .rename(columns={"duration": "observated_times"})
    )
    sum_per_uuid = observated_matrix.groupby("rank")["observated_times"].transform(
        "sum"
    )
    observated_matrix["Prob"] = observated_matrix["observated_times"] / sum_per_uuid
    Prob_matrix = observated_matrix.pivot_table(
        index="rank", columns="gridID", values="Prob"
    ).fillna(0)
    return Prob_matrix


# TODO Efficiency!
# calculating PSI
def calculate_psi_csr(x, social_similarity):
    y = x.copy()
    # fix here
    df_np = social_similarity.to_numpy()
    for i, val_y in x.iteritems():
        val = 0
        val2 = 0
        flag = 0
        if val_y != 0:
            for j, val_x in x.iteritems():
                if val_x != 0 and i != j:
                    flag = 1
                    val += df_np[i - 1][j - 1] * val_x
                    val2 += val_x
            if flag == 1:
                y.at[i] = val / val2
            else:
                y.at[i] = 1
    #     print("1")
    return y


def process_chunk(start_col, end_col, prob_matrix, social_similarity):
    PSI_individual_location_matrix_chunk = [
        calculate_psi_csr(prob_matrix.iloc[:, col], social_similarity)
        for col in range(start_col, end_col)
    ]
    return np.column_stack(PSI_individual_location_matrix_chunk)

# getting the PSI matrix


def get_PSI_individual_location_matrix_chunked(
    prob_matrix, chunk_size=500, social_similarity=None
):
    print(f"size:{prob_matrix.shape}")
    n_cols = prob_matrix.shape[1]
    result_chunks = []

    with ProcessPoolExecutor() as executor:
        futures = []

        for start_col in range(0, n_cols, chunk_size):
            end_col = min(start_col + chunk_size, n_cols)
            future = executor.submit(
                process_chunk, start_col, end_col, prob_matrix, social_similarity
            )
            futures.append((start_col, future))

        for start_col, future in tqdm(
            sorted(futures, key=lambda x: x[0]),
            desc="Processing chunks",
            total=len(futures),
        ):
            result_chunks.append(future.result())

    PSI_individual_location_matrix_np = np.column_stack(result_chunks)

    PSI_individual_location_matrix = pd.DataFrame(
        PSI_individual_location_matrix_np,
        index=prob_matrix.index,
        columns=prob_matrix.columns,
    )

    return PSI_individual_location_matrix


def get_PSI_individual_location_matrix(Prob_matrix, social_similarity, timewindow=None):
    PSI_individual_location_matrix = get_PSI_individual_location_matrix_chunked(
        Prob_matrix, 400, social_similarity
    )
    if not time:
        PSI_individual_location_matrix.to_csv(
            rf'PSI_individual_location_matrix_{time.strftime("%Y%m%d")}.csv',
            encoding="utf-8_sig",
        )
    else:
        PSI_individual_location_matrix.to_csv(
            rf'PSI_individual_location_matrix_timewindow_{timewindow}_{time.strftime("%Y%m%d")}.csv',
            encoding="utf-8_sig",
        )
    return PSI_individual_location_matrix

# PSI matrix group by individual


def get_PSI_matrix_for_individual(PSI_individual_location_matrix, prob_matrix):
    PSI_x_matrix = pd.DataFrame(
        PSI_individual_location_matrix.values * prob_matrix.values,
        columns=PSI_individual_location_matrix.columns,
        index=PSI_individual_location_matrix.index,
    )
    PSI_Individual = pd.DataFrame(PSI_x_matrix.sum(axis=1), columns=["PSI"])
    return PSI_Individual

# PSI matrix group by spatial unit


def get_PSI_matrix_for_unit(grids, PSI_individual_location_matrix, prob_matrix):
    try:
        grids["gridID"] = grids["gridID"].astype(str)
        PSI_x_matrix = pd.DataFrame(
            PSI_individual_location_matrix.values * prob_matrix.values,
            columns=PSI_individual_location_matrix.columns,
            index=PSI_individual_location_matrix.index,
        )

        PSI_grid = gpd.GeoDataFrame(
            (
                pd.DataFrame(
                    PSI_x_matrix.sum().values / prob_matrix.sum().values,
                    columns=["PSI"],
                    index=PSI_individual_location_matrix.columns,
                ).merge(grids, left_index=True, right_on="gridID")
            ),
            geometry="geometry",
            crs="epsg:4326",
        )
    except:
        grids["gridID"] = grids["gridID"].astype("int64")
        PSI_x_matrix = pd.DataFrame(
            PSI_individual_location_matrix.values * prob_matrix.values,
            columns=PSI_individual_location_matrix.columns,
            index=PSI_individual_location_matrix.index,
        )

        PSI_grid = gpd.GeoDataFrame(
            (
                pd.DataFrame(
                    PSI_x_matrix.sum().values / prob_matrix.sum().values,
                    columns=["PSI"],
                    index=PSI_individual_location_matrix.columns,
                ).merge(grids, left_index=True, right_on="gridID")
            ),
            geometry="geometry",
            crs="epsg:4326",
        )
    return PSI_grid

# PSI matrix group by time window


def get_prob_matrix_in_timewindow(stay_df, time, grids, RANK):
    stay_df["timewindow"] = stay_df["stime"].dt.hour
    stay_df = stay_df[stay_df["timewindow"] == time]
    grids["gridID"] = grids["gridID"].astype("int64")
    stay_df_merged = stay_df.merge(
        grids[["LONCOL", "LATCOL", "gridID"]], on=["LONCOL", "LATCOL"]
    ).merge(RANK, left_on="uuid", right_on="uuid")[
        ["uuid", "gridID", "rank", "duration"]
    ]
    observated_matrix = (
        stay_df_merged.groupby(["rank", "gridID"])
        .count()[["duration"]]
        .rename(columns={"duration": "observated_times"})
    )
    sum_per_uuid = observated_matrix.groupby("rank")["observated_times"].transform(
        "sum"
    )
    observated_matrix["Prob"] = observated_matrix["observated_times"] / sum_per_uuid
    Prob_matrix = observated_matrix.pivot_table(
        index="rank", columns="gridID", values="Prob"
    ).fillna(0)
    return Prob_matrix


def get_timewindow_average_psi(stay_df, grids, RANK, root):
    psi = []
    for tw in range(24):
        window = pd.read_csv(
            rf"{root}/PSI_individual_location_matrix_timewindow_{tw}_20230411.csv",
            index_col=[0],
        )
        Prob_matrix = get_prob_matrix_in_timewindow(stay_df, tw, grids, RANK)
        PSI_Individual = get_PSI_matrix_for_individual(window, Prob_matrix)
        psi.append(PSI_Individual.mean().values[0])
    return psi


def get_timewindow_unit_psi(stay_df, grids, RANK, root):
    psi = []
    for tw in range(24):
        window = pd.read_csv(
            rf"{root}/PSI_individual_location_matrix_timewindow_{tw}_20230411.csv",
            index_col=[0],
        )
        Prob_matrix = get_prob_matrix_in_timewindow(stay_df, tw, grids, RANK)
        PSI_unit = get_PSI_matrix_for_unit(grids, window, Prob_matrix)
        grids.merge(PSI_unit, on="gridID", how="left", suffixes=("", "_y")).drop(
            ["LONCOL_y", "LATCOL_y", "geometry_y"], axis=1
        ).fillna(0).to_csv(
            rf"./result/TIME_WINDOW/PSI_UNIT/psi_unit_{tw}.csv", index=False
        )
    return psi


def read_bson(bson_file):
    data = bson.decode_file_iter(open(bson_file, "rb"))
    df = pd.DataFrame(data)
    df["geometry"] = gpd.points_from_xy(df["lng"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    return gdf


def process_fact_yearly(grids, year):
    import json

    for i in range(year, year + 1):
        with open(rf"./data/grid_factories/raw_json/fact{i}.json", "r") as file:
            data = json.load(file)

        df = pd.DataFrame(data["gridhots"])
        df["x"] = df["x"].astype("int64")
        df["y"] = df["y"].astype("int64")
        df["x"], df["y"] = tbd.bd09mctobd09(df["x"], df["y"])
        df["x"], df["y"] = tbd.bd09towgs84(df["x"], df["y"])
        df["geometry"] = gpd.points_from_xy(df["x"], df["y"])
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
        year = str(i)
        fact = gdf
        factoreis_count = (
            gpd.sjoin(fact, grids, predicate="within")
            .groupby("gridID")
            .count()[["num"]]
        )
        grids["fact" + year] = pd.merge(
            grids, factoreis_count, left_on="gridID", right_index=True, how="left"
        ).fillna(0)["num"]
    return gdf


def process19data():
    phone = pd.read_csv(
        r"./data/phones/20190115.csv",
        header=None,
        encoding="gbk",
        names=["date", "time", "s", "e", "age", "volumn"],
    )
    phone_grid = gpd.read_file(
        r"./data/boundary/shenzhen_net/shenzhen_net84.geojson")
    phone = phone[phone["time"].isin([6, 7, 8, 9])]
    stat = phone.groupby(["s", "e"]).count()[["volumn"]].reset_index()
    stat_geo = (
        stat.merge(phone_grid[["Tid", "geometry"]],
                   left_on="s", right_on="Tid")
        .rename(columns={"geometry": "start_geometry"})
        .merge(phone_grid[["Tid", "geometry"]], left_on="e", right_on="Tid")
        .rename(columns={"geometry": "end_geometry"})
        .drop(["Tid_x", "Tid_y"], axis=1)
    )
    stat_geo["start_geometry"], stat_geo["end_geometry"] = stat_geo[
        "start_geometry"
    ].apply(lambda x: x.centroid), stat_geo["end_geometry"].apply(lambda x: x.centroid)
    stat_geo["mark"] = stat_geo.index
    stat_geo_new = stat_geo.iloc[stat_geo.index.repeat(
        2)].reset_index(drop=True)
    stat_geo_new["ind"] = stat_geo_new.index
    stat_geo_new["geometry"] = stat_geo_new.apply(
        lambda x: x["start_geometry"] if x["ind"] % 2 == 0 else x["end_geometry"],
        axis=1,
    )
    stat_geo = stat_geo_new[["mark", "volumn", "geometry"]]
    stat_geo["home"] = stat_geo.index
    stat_geo["home"] = stat_geo.apply(
        lambda x: True if x["home"] % 2 == 0 else False, axis=1
    )
    stat_geo = stat_geo.rename(columns={"mark": "uuid"})
    stat_geo

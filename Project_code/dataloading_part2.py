import pandas as pd
import numpy as np
from constants import *

def data_loading(df: pd.DataFrame, data_source):
    col_names = list(df.columns)

    # create the X data
    n_rows = len(df.values[:, 0])
    # X_col_names = [attr_name for attr_name in col_names if attr_name != y_attr_name]
    attr_to_X_col_idx = {}
    X_cols = []

    if data_source == "df_series":
        # adding the columns with same structure:
        for attr in [runtime_name, startYear_name, endYear_name, durationYears_name,
                     nEpisodes_name, averageRating_name, numVotes_name]:
            attr_to_X_col_idx[attr] = len(X_cols)
            col_data = np.asarray(df.values[:, col_names.index(attr)], dtype=str)
            X_cols.append(col_data)

    elif data_source == "df_movies":
        # adding the columns with same structure:
        for attr in [runtime_name, startYear_name, averageRating_name, numVotes_name,
                     tconst_name]:
            attr_to_X_col_idx[attr] = len(X_cols)
            col_data = np.asarray(df.values[:, col_names.index(attr)], dtype=str)
            X_cols.append(col_data)

    elif data_source == "df_movies_extra":
        for attr in [movie_popularity_name, cast_popularity_name, nUser_reviews_name,
                     gross_name, nCritic_reviews_name, budget_name, movie_link_name]:
            attr_to_X_col_idx[attr] = len(X_cols)
            col_data = np.asarray(df.values[:, col_names.index(attr)], dtype=str)
            X_cols.append(col_data)

    X = np.vstack(X_cols).T
    X = np.delete(X, np.where(X == "nan")[0], axis=0)

    return X, attr_to_X_col_idx
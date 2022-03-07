# this script is used for doing the PCA and displaying it using plots
import pandas as pd
import numpy as np
from constants import runtime_name, startYear_name, endYear_name, durationYears_name, nEpisodes_name, averageRating_name, \
    numVotes_name, genres_name


def data_loading(df: pd.DataFrame,y_attr_name):
    col_names = list(df.columns)
    # extract our y data, the averageRating:
    y_col_idx = col_names.index(y_attr_name)
    y = df.values[:, y_col_idx]

    # create the X data
    n_rows = len(df.values[:,0])
    X_col_names = [attr_name for attr_name in col_names if attr_name != y_attr_name]
    attr_to_X_col_idx = {}
    X_cols = []


    # adding the columns with same structure:
    for attr in [runtime_name, startYear_name, endYear_name, durationYears_name, nEpisodes_name, averageRating_name,
                 numVotes_name]:
        if attr in X_col_names:
            attr_to_X_col_idx[attr] = len(X_cols)
            X_cols.append(df.values[:, col_names.index(attr)])


    # encoding the genres:
    if genres_name in X_col_names:
        # So far, genres are stored as comma-separated strings in the same column.
        # We want to encode genres to be K binary columns, with K = number of different genres
        # if a row is listed with genre g1, ..., gn, then the value for those rows will be 1, and 0 for any other genre columns
        # first, figure out K:
        genres_data = df.values[:, col_names.index(genres_name)]
        genre_list = []
        for row in genres_data:
            genre_list += row.split(",")
        unique_genres = list(set(genre_list))
        K = len(unique_genres)

        genre_col = np.empty(n_rows)
        for i, genre in enumerate(unique_genres):
            for row_i,row_genres in enumerate(genres_data):
                genre_col[row_i] = int(genre in row_genres) # 0 or 1
            # attr_to_X_col_idx[genre] = len(X_cols)
            # X_cols.append(genre_col)
        X = np.vstack(X_cols)
        return y, X

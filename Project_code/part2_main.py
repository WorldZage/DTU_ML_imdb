# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import summary_statistics as su
import data_generator as dg
import dataloading_part2 as dl2
from constants import *


# import apply_ex5


def write_filtered_and_movie_metadata_to_file():
    dataset_path_n_parents = "../../../"
    # dg.count_valid_rows()
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv/data.tsv"
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    dg.num_ratings_filter(ds, min_n_votes=1000)
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv/data.tsv"
    dg.extend_by_file_and_tconst(ds, basics_path, [titleType_name, genres_name, startYear_name, runtime_name])
    dg.title_type_filter(ds, only_title_type="movie")
    print(len(ds.data_map))

    # Extending by the movie metadata file
    metadata_path = dataset_path_n_parents + "datasets/movie_metadata.csv"
    metadata_attrs = [movie_popularity_name, cast_popularity_name, nUser_reviews_name, gross_name,
                      nCritic_reviews_name, budget_name, cast_pop_1, cast_pop_2]
    dg.extend_by_metadata_file(ds, metadata_path, metadata_attrs)
    print(f"{ds.attributes = }")
    before_missing_filter = len(ds.data_map)

    dg.missing_data_filter(ds)
    after_missing_filter = len(ds.data_map)
    diff_n_rows = before_missing_filter - after_missing_filter
    print(f"{before_missing_filter = }, {after_missing_filter = }\n {diff_n_rows = }")

    desired_attributes = [tconst_name, titleType_name, genres_name, runtime_name, startYear_name,
                          averageRating_name, numVotes_name] + metadata_attrs
    ds.write_to_file("collected_movie_data.csv", desired_attributes)


if __name__ == '__main__':

    """ Movie data part:"""
    # write_filtered_and_movie_metadata_to_file()
    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    data, col_idx_dict = dl2.data_loading(df_movies, "df_movies_and_extra")
    data = np.array(data, dtype=float)


    # Extract the X and y data
    y = data[col_idx_dict[averageRating_name]]
    X1 = data[:col_idx_dict[averageRating_name]]
    X2 = data[col_idx_dict[averageRating_name]+1:]
    X = np.vstack(X1,X2)

    # Get the dimensions of the data
    N, M = X.shape

    # Transform the data to be centered:
    X_center = X - np.ones((N, 1)) * X.mean(axis=0)
    # Transform the data to be normalized
    X_norm = X_center * (1 / np.std(X_center, axis=0))

    cor_mat = np.corrcoef(X.T).round(3)
    print(f"{col_idx_dict}")
    print(cor_mat)
    """df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    X, col_idx_dict = dl2.data_loading(df_movies, "df_movies")
    X_df1 = X
    attr_dict_df1 = col_idx_dict

    df_movies_extra = pd.read_csv("movie_metadata.csv", sep=",", dtype=str)

    X, col_idx_dict = dl2.data_loading(df_movies_extra, "df_movies_extra")
    X_df2 = X
    attr_dict_df2 = col_idx_dict

    X_conc = np.array(0)
    for idx1 in range(len(X_df1)):
        movie_id = X_df1[idx1, attr_dict_df1[tconst_name]]
        for idx2 in range(len(X_df2)):
            movie_link = X_df2[idx2, attr_dict_df2[movie_link_name]]
            if movie_id in movie_link:
                X_conc = np.append(X_conc, X_df1[idx1, :])
                X_conc = np.append(X_conc, X_df2[idx2, :])

    nrows1, ncols1 = X_df1.shape
    nrows2, ncols2 = X_df2.shape

    X_conc = np.delete(X_conc, 0)
    X_conc = X_conc.reshape(int(len(X_conc) / (ncols1 + ncols2)), ncols1 + ncols2)

    X_conc = np.delete(X_conc, [attr_dict_df1[tconst_name], attr_dict_df2[movie_link_name] + ncols1], 1)

    # force to be float
    X_conc = np.array(X_conc, dtype=float)

    # normalize
    X_norm = X_conc * (1 / np.std(X_conc, axis=0))
    cov_mat = np.cov(X_norm.T).round(3)
    print(f"cov.mat: {cov_mat}")

    attr_dict_conc = np.concatenate((list(attr_dict_df1.keys()), list(attr_dict_df2.keys())))
    # attr_cov = np.delete(attr_dict_conc, [attr_dict_df1[tconst],attr_dict_df2[movie_link_name]+ncols1], 0)
    # attr_cov = attr_cov.T

    X_highrates = np.delete(X_conc, np.where(X_conc[:, 2] < 7)[0], axis=0)

    X = np.delete(X_highrates, [attr_dict_df1[averageRating_name]], 1)
    y = X_highrates[:, attr_dict_df1[averageRating_name]]

    # take out 1-startYear; 4-movie_facebook_likes; 9-budget
    X = np.delete(X, [1, 4, 8], 1)
    # apply_ex5.linear_regression(y, X)
    """

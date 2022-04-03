# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import data_generator as dg
import summary_statistics as su
import dataloading_part2 as dl
import pca_and_plots as pca
from constants import *
# import apply_ex5


def write_filtered_movie_data_to_file():
    dataset_path_n_parents = "../../"
    # dg.count_valid_rows()
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv"  # /data.tsv
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.ratings_filter(1000))
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv"  # /data.tsv
    ds.run_func_on_ds(dg.extend_by_file_and_tconst(basics_path,
                                                   ["titleType", "genres", "isAdult", "startYear", "runtimeMinutes"]))
    ds.run_func_on_ds(dg.title_type_filter("movie"))
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.missing_data_filter())
    ds.write_to_file("collected_movie_data.csv",
                     ["tconst", "titleType", "genres", "runtimeMinutes", "startYear", "isAdult", "averageRating",
                      "numVotes"])


def write_filtered_tvseries_data_to_file():
    dataset_path_n_parents = "../../"
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv"  # /data.tsv
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.ratings_filter(1000))
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv"  # /data.tsv
    ds.run_func_on_ds(dg.extend_by_file_and_tconst(basics_path,
                                                   ["titleType", "genres", "isAdult", "startYear", "endYear",
                                                    "runtimeMinutes"]))
    ds.run_func_on_ds(dg.title_type_filter("tvSeries"))
    print(len(ds.data_map))

    episode_path = dataset_path_n_parents + "datasets/title.episode.tsv/data.tsv"  #
    ds.run_func_on_ds(dg.extend_n_episodes(episode_path))

    ds.run_func_on_ds(dg.extend_show_duration())

    ds.run_func_on_ds(dg.missing_data_filter())

    ds.write_to_file("collected_tvseries_data.csv",
                     ["tconst", "titleType", "genres", "runtimeMinutes", "startYear", "endYear", "durationYears",
                      "nEpisodes", "isAdult", "averageRating",
                      "numVotes"])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    """ Movie data part:"""
    # write_filtered_movie_data_to_file()
    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    # summ_stats = su.calculate_summary_stats(df_movies, ["runtimeMinutes", "startYear", "averageRating", "numVotes"])
    # for attr in summ_stats:
    #    print(attr)

    """ Series data part:"""
    # write_filtered_tvseries_data_to_file()
    df_series = pd.read_csv("collected_tvseries_data.csv", sep="\t", dtype=str)
    summ_stats = su.calculate_summary_stats(df_series, ["runtimeMinutes", "startYear",
                                                        "endYear", "durationYears", "nEpisodes",
                                                        "averageRating", "numVotes"])
    # for attr in summ_stats:
    #     print(attr)

    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    X, col_idx_dict = dl.data_loading(df_movies, "df_movies")
    X_df1 = X
    attr_dict_df1 = col_idx_dict

    df_movies_extra = pd.read_csv("movie_metadata.csv", sep=",", dtype=str)

    X, col_idx_dict = dl.data_loading(df_movies_extra, "df_movies_extra")
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




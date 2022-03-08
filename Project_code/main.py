# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import data_generator as dg
import summary_statistics as su
import pca_and_plots as pca
from constants import *


def write_filtered_movie_data_to_file():
    dataset_path_n_parents = "../../../"
    # dg.count_valid_rows()
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv/data.tsv"
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.ratings_filter(1000))
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv/data.tsv"
    ds.run_func_on_ds(dg.extend_by_file_and_tconst(basics_path,
                                                   ["titleType", "genres", "isAdult", "startYear", "runtimeMinutes"]))
    ds.run_func_on_ds(dg.title_type_filter("movie"))
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.missing_data_filter())
    ds.write_to_file("collected_movie_data.csv",
                     ["tconst", "titleType", "genres", "runtimeMinutes", "startYear", "isAdult", "averageRating",
                      "numVotes"])


def write_filtered_tvseries_data_to_file():
    dataset_path_n_parents = "../../../"
    rating_path = dataset_path_n_parents + "datasets/title.ratings.tsv/data.tsv"
    ds = dg.DataSet(rating_path)
    print(len(ds.data_map))

    ds.run_func_on_ds(dg.ratings_filter(1000))
    print(len(ds.data_map))

    basics_path = dataset_path_n_parents + "datasets/title.basics.tsv/data.tsv"
    ds.run_func_on_ds(dg.extend_by_file_and_tconst(basics_path,
                                                   ["titleType", "genres", "isAdult", "startYear", "endYear", "runtimeMinutes"]))
    ds.run_func_on_ds(dg.title_type_filter("tvSeries"))
    print(len(ds.data_map))

    episode_path = dataset_path_n_parents + "datasets/title.episode.tsv/data.tsv"
    ds.run_func_on_ds(dg.extend_n_episodes(episode_path))

    ds.run_func_on_ds(dg.extend_show_duration())

    ds.run_func_on_ds(dg.missing_data_filter())

    ds.write_to_file("collected_tvseries_data.csv",
                     ["tconst", "titleType", "genres", "runtimeMinutes", "startYear", "endYear", "durationYears", "nEpisodes", "isAdult", "averageRating",
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
    summ_stats = su.calculate_summary_stats(df_series, ["runtimeMinutes", "startYear", "endYear", "durationYears", "nEpisodes",
                               "averageRating", "numVotes"])
    for attr in summ_stats:
        print(attr)

    y, X, col_idx_dict = pca.data_loading(df_series, averageRating_name)
    avgRate_data = y
    x_data = X[col_idx_dict[endYear_name]]
    print(col_idx_dict.items())
    # pca.visualize(x_data, avgRate_data)
    all_numerical = np.vstack((y,X))
    all_numerical = all_numerical.T
    # normalize
    Y = all_numerical * (1/np.std(all_numerical,axis=0))
    np.set_printoptions(formatter={'all':lambda x: str(x)})
    cov_mat = np.cov(Y.T).round(3)

    print(f"cov.mat: {cov_mat}")
    #pca.PCA(all_numerical)
    attr_names = ["averageRating"] + list(col_idx_dict.keys())
    #pca.PCA_bar_plot(all_numerical, attr_names)

    #pca.norm_plots(all_numerical.T, attr_names)
    pca.project_plot(y,all_numerical)
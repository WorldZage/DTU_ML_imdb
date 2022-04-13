# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
from matplotlib.pyplot import plot, show, title, figure, subplot, xlabel, ylabel, subplots_adjust, subplots, ylim

import summary_statistics as su
import data_generator as dg
import dataloading_part2 as dl2
from constants import *

# import apply_ex5
from cross_validation import cross_validate_lambda, cross_validate_feature


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


def regression_a(df: pd.DataFrame):
    data, col_idx_dict = dl2.data_loading(df, "df_movies_and_extra")
    data = np.array(data, dtype=float)

    cor_mat = np.corrcoef(data.T).round(3)
    print(f"{col_idx_dict}")
    print(cor_mat[:, 2])


    # Extract the X and y data
    y_col_idx = col_idx_dict[averageRating_name]
    y = data[:, y_col_idx]
    X = np.delete(data, y_col_idx, 1)
    feature_names = [''] * len(col_idx_dict)

    # Data pre processing
    for name, idx in col_idx_dict.items():
        feature_names[idx] = name
    # Remove the y column's name from the feature names:
    del feature_names[feature_names.index(averageRating_name)]

    startYear_idx = col_idx_dict[startYear_name]
    # By advice of TA, we do a (superficial) inspection of the relation between year and average rating:
    def median_rating_per_year_plot():
        min_startYear = int(np.min(X[:,startYear_idx]))
        max_startYear = int(np.max(X[:,startYear_idx]))

        # a list containing non-duplicates of the year entries
        years = np.unique(X[:,startYear_idx])
        # empty vector to be filled later
        median_rating_per_year = np.zeros([years.size, 2])
        # Iterate through the set of years, for each year calculate its corresponding median averageRating.
        for idx, year in enumerate(years):
            median_rating_per_year[idx,0] = year
            median_rating_per_year[idx,1] = np.median([y[X[:, startYear_idx] == year]])
        fig, (ax1, ax2) = subplots(1, 2, sharey=True)
        ax1.plot(median_rating_per_year[:,0],median_rating_per_year[:,1], '+')
        fig.supylabel("median rating")
        ax1.title.set_text("The median of 'averageRating' per year, of movies on IMDB")
        fig.supxlabel("Year")
        # plotting a zoomed in version of the data, for years 1996-2016 ( most recent years in the data)
        last_20_years_data = ((median_rating_per_year[:,0][median_rating_per_year[:,0] > max_startYear - 20].round(0)),
                              median_rating_per_year[:,1][median_rating_per_year[:,0] > max_startYear - 20])
        ax2.plot(last_20_years_data[0],last_20_years_data[1], '+',color="red")
        ax2.title.set_text("The median of 'averageRating' per year, of movies on IMDB, in the last 25 years.")
        ylim(0,10)
        show()
    # median_rating_per_year_plot()

    # it's difficult to imagine a linear relation between starting year, and movie rate.
    # Some periods of time might have higher rated movies than others.
    # Overall we might see an average tendency to decrease or increase in rating, but it won't help us with predictions.
    # So we remove it from our dataset.

    X = np.delete(X, startYear_idx, 1)
    del feature_names[feature_names.index(startYear_name)]

    # Some features are spread across a large scale values.
    # For example, some movies have several thousand likes, while others only have tens.
    # to make them comparable, we use logarithmic transformation on popularity features
    # (but we will have to remove rows that contain 0 in any of these attributes)
    log_feature_idcs = [feature_names.index(name) for name in
                        [numVotes_name, movie_popularity_name, cast_popularity_name, nUser_reviews_name,
                         nCritic_reviews_name]]
    # Removing the rows containing 0, from both X and y data:
    X_no_0 = X[(X[:, log_feature_idcs] > 0.00).all(axis=1)]
    y_no_0 = y[(X[:, log_feature_idcs] > 0.00).all(axis=1)]

    # Logarithmic transformation
    X_log = X_no_0
    X_log[:, log_feature_idcs] = np.log(X_no_0[:, log_feature_idcs])

    # check new correlations
    # stacking the y data and x data together
    cor_data = np.hstack((np.array([y_no_0]).T, X_log))
    cor_mat = np.corrcoef(cor_data.T).round(3)
    print(f"NEW correlation matrix:\n{cor_mat[:, 0]}")
    print(f"{[averageRating_name] + feature_names}")

    # Get the dimensions of the data
    N, M = X_log.shape
    print(f"{N = }, {M = }")

    # Transform the data to be centered:
    X_center = X_log - np.ones((N, 1)) * X_log.mean(axis=0)
    # Transform the data to be normalized
    X_norm = X_center * (1 / np.std(X_center, axis=0))
    # print(f"{X_norm[0:3, :]}")

    # Feature selection:
    # cross_validate_feature(X_norm, y_no_0, 10, feature_names)
    # Based off the results, we will now remove "budget", "gross" and "num_critic_for_reviews" from the features:
    removed_feature_idcs = [feature_names.index(budget_name),
                            feature_names.index(gross_name),
                            feature_names.index(nCritic_reviews_name)]
    X_feat_selected = np.delete(X_norm, [removed_feature_idcs], 1)
    for idx in sorted(removed_feature_idcs)[::-1]:
        # traverse backwards to avoid the problem of deleting from an array that is being iterated through
        del feature_names[idx]
    print(f"{feature_names = }, {X_feat_selected.shape = }")
    lambdas = np.power(10.,np.arange(-5,5,1))
    #np.array([10 ** -5, 10 ** -3, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4])
    # cross_validate_lambda(X_feat_selected, y_no_0, 10, feature_names, lambdas)


if __name__ == '__main__':
    """ Movie data part:"""
    # write_filtered_and_movie_metadata_to_file()
    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    regression_a(df_movies)

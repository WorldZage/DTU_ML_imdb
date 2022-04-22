# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pprint import pprint

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import tree
from platform import system
from os import getcwd

from matplotlib.pyplot import plot, show, title, figure, subplot, xlabel, ylabel, subplots_adjust, subplots, ylim, hist, legend, imread
from scipy.stats import stats

from toolbox_02450 import windows_graphviz_call

import summary_statistics as su
import data_generator as dg
import dataloading_part2 as dl2
import dataloading_part2_1 as dl3

from constants import *

# import apply_ex5
from cross_validation import cross_validate_lambda, cross_validate_feature, cross_validate_ann, \
    optimal_hidden_unit_ann, cross_validate_model_comparison, statistic_comparison, plot_models

from classification import compare_models, logistic_reg

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

    show_correlations = False
    if show_correlations:
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
        min_startYear = int(np.min(X[:, startYear_idx]))
        max_startYear = int(np.max(X[:, startYear_idx]))

        # a list containing non-duplicates of the year entries
        years = np.unique(X[:, startYear_idx])
        # empty vector to be filled later
        median_rating_per_year = np.zeros([years.size, 2])
        # Iterate through the set of years, for each year calculate its corresponding median averageRating.
        for idx, year in enumerate(years):
            median_rating_per_year[idx, 0] = year
            median_rating_per_year[idx, 1] = np.median([y[X[:, startYear_idx] == year]])
        fig, (ax1, ax2) = subplots(1, 2, sharey=True)
        ax1.plot(median_rating_per_year[:, 0], median_rating_per_year[:, 1], '+')
        fig.supylabel("median rating")
        ax1.title.set_text("The median of 'averageRating' per year, of movies on IMDB")
        fig.supxlabel("Year")
        # plotting a zoomed in version of the data, for years 1996-2016 ( most recent years in the data)
        last_20_years_data = (
            (median_rating_per_year[:, 0][median_rating_per_year[:, 0] > max_startYear - 20].round(0)),
            median_rating_per_year[:, 1][median_rating_per_year[:, 0] > max_startYear - 20])
        ax2.plot(last_20_years_data[0], last_20_years_data[1], '+', color="red")
        ax2.title.set_text("The median of 'averageRating' per year, of movies on IMDB, in the last 25 years.")
        ylim(0, 10)
        show()

    # median_rating_per_year_plot()

    # it's difficult to imagine a linear relation between starting year, and movie rate.
    # Some periods of time might have higher rated movies than others.
    # Overall we might see an average tendency to decrease or increase in rating, but it won't help us with predictions.
    # So we remove it from our dataset.

    X = np.delete(X, startYear_idx, 1)
    del feature_names[feature_names.index(startYear_name)]

    # Some features are spread across a large scale of values.
    # For example, some movies have several thousand likes, while others only have tens.
    # to make them comparable, we use logarithmic transformation on popularity features
    # (but we will have to remove rows that contain 0 in any of these attributes)
    log_features = [numVotes_name, movie_popularity_name, cast_popularity_name, nUser_reviews_name,
                    nCritic_reviews_name]
    log_feature_idcs = [feature_names.index(name) for name in
                        log_features]
    # Removing the rows containing 0, from both X and y data:
    X_no_0 = X[(X[:, log_feature_idcs] > 0.00).all(axis=1)]
    y_no_0 = y[(X[:, log_feature_idcs] > 0.00).all(axis=1)]

    # Logarithmic transformation
    X_log = X_no_0
    X_log[:, log_feature_idcs] = np.log(X_no_0[:, log_feature_idcs])
    # Change the name of the transformed features:
    feature_names = ["log({})".format(name) if name in log_features else name
                     for name in feature_names]

    if show_correlations:
        # check new correlations
        # stacking the y data and x data together
        cor_data = np.hstack((np.array([y_no_0]).T, X_log))
        cor_mat = np.corrcoef(cor_data.T).round(3)
        print(f"NEW correlation matrix:\n{cor_mat[:, 0]}")
        print(f"{[averageRating_name] + feature_names}")

    # Get the dimensions of the data
    N, M = X_log.shape

    # Transform the data to be centered:
    X_center = X_log - np.ones((N, 1)) * X_log.mean(axis=0)
    # Transform the data to be normalized
    X_norm = X_center * (1 / np.std(X_center, axis=0))

    show_mean_and_std = True
    if show_mean_and_std:
        print(f"MEANS: {X_log.mean(axis=0).round(2)}\n"
              f"St.D's: {(np.std(X_center, axis=0).round(2))}\n"
              f"FEATURES: {list(enumerate(feature_names))})")

    # Feature selection:
    if False:  # Change to True if you want to do the feature selection again.
        cross_validate_feature(X_norm, y_no_0, 10, feature_names)
    # Based off the results, we will now remove "budget", "gross" and "num_critic_for_reviews" from the features:
    removed_feature_idcs = [feature_names.index(budget_name),
                            feature_names.index(gross_name),
                            feature_names.index("log({})".format(nCritic_reviews_name))]
    X_feat_selected = np.delete(X_norm, [removed_feature_idcs], 1)
    for idx in sorted(removed_feature_idcs)[::-1]:
        # traverse backwards to avoid the problem of deleting from an array that is being iterated through
        del feature_names[idx]
    print(f"{feature_names = }, {X_feat_selected.shape = }")
    lambdas = np.power(10., np.arange(-15, 4, 2))
    print(f"{lambdas = }")
    # np.array([10 ** -5, 10 ** -3, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4])
    # cross_validate_lambda(X_feat_selected, y_no_0, 10, feature_names, lambdas)
    return X_feat_selected, y_no_0, feature_names


def regression_b(X, y, feature_names):
    do_pca_preprocessing = False
    if do_pca_preprocessing:
        Y = stats.zscore(X, 0)
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        V = V.T
        # Components to be included as features
        k_pca = 3
        X = X @ V[:, :k_pca]
        N, M = X.shape
        feature_names = ["PC {}".format(i) for i in range(3)]
    # cross_validate_ann(X, y, 5, feature_names)
    lambdas = np.power(10., np.arange(-3, 3, 1))
    hidden_unit_options = [16, 18, 20, 22, 24]  # [1, 20]
    # [5, 10, 20, 30] <-- Actual options used in the report
    """
    xt = np.array([np.random.random_integers(0,20,10),np.random.random_integers(10,40,10)])
    yt = np.array(np.random.random_integers(100,110,10))
    """
    K = 5
    # print(optimal_hidden_unit_ann(X, y, hidden_unit_options, cvf=5))
    if False:
        model_comparison_dict = (cross_validate_model_comparison(X, y, lambdas, hidden_unit_options, K=K))
        pprint(model_comparison_dict)

        # --
        # RLR
        rlr_result = model_comparison_dict["RLR"]
        opt_lambdas_lst = list(zip(*rlr_result))[0]
        opt_lambdas_lst = [lamb for lamb in opt_lambdas_lst]
        # ANN
        ann_result = model_comparison_dict["ANN"]
        opt_hu_lst = list(zip(*ann_result))[0]
        opt_hu_lst = [hu for hu in opt_hu_lst]

        def most_frequent(input_lst):
            # https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
            return max(set(input_lst), key=input_lst.count)

        opt_lambda = most_frequent(opt_lambdas_lst)
        opt_hu = most_frequent(opt_hu_lst)

        print(f"{opt_hu = }\n"
              f"{opt_lambda = }")
    K = 2
    opt_hu = 24
    opt_lambda = 1
    statistic_comparison(X, y, opt_hu, opt_lambda, K=K)
    plot_models(X, y, 1, 20, K=2)


def classification_models(data, col_idx_dict):
    
    N, M = data.shape
    
    # Extract the X and y data
    y_col_idx = col_idx_dict[averageRating_name]
    
    # order descending considering average rating
    
    data_ordered = data[data[:, y_col_idx].argsort()[::-1]]
    
    # BALANCE DATA
    rate_class_limit = 7.5
    count_highrated = np.count_nonzero(data[:,y_col_idx] >= rate_class_limit)
    print(count_highrated)

    data_highrated =  data_ordered[:count_highrated, :]
    data_lowrated = data_ordered[count_highrated:, :]
    
    print(data.shape)
    print(data_highrated.shape)
    print(data_lowrated.shape)
    
    # solve undersample of high rated
    #data_highrated = np.tile(data_highrated,(2,1))
    
    np.random.shuffle(data_lowrated)
    
    limit_sample = len(data_highrated)
    data_lowrated_sample = np.empty((limit_sample,M))
    for i in range(limit_sample):
        data_lowrated_sample[i,:] = data_lowrated[i, :]
    
    data_filt = np.concatenate((data_highrated, data_lowrated_sample), axis=0)

    # remove y + uncorrelated attirbutes from data
    X = np.delete(data_filt, [y_col_idx, 
                         col_idx_dict[startYear_name], 
                         col_idx_dict[cast_popularity_name], 
                         col_idx_dict[budget_name],
                         col_idx_dict["gross"]
                         ], 1)

    y = data_filt[:, y_col_idx]
    y_binary = y.copy();
    for i in range(len(y)):
        if y[i] >= rate_class_limit:
            y_binary[i]=0
        else:
            y_binary[i]=1
    
    # CLASSIFY
    logistic_reg(X, y_binary, rate_class_limit)
    
    # COMPARY CLASSIFIERS
    compare_models(X, y_binary)
    

    return X, y_binary, data_filt

if __name__ == '__main__':
    """ Movie data part :"""
    # write_filtered_and_movie_metadata_to_file()
    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)
    X, y, feats = regression_a(df_movies)

    # regression_b(X, y, feats)
    
    # convert to array
    data, col_idx_dict, col_idx_arr = dl3.data_loading(df_movies, "df_movies_and_extra")
    data = np.array(data, dtype=float)
    
    #normalize
    data_norm = data * (1/np.std(data,axis=0))
    cov_mat = np.cov(data_norm.T).round(3)
    #print(f"cov.mat: {cov_mat}")
    
    # investigate linear regression
    #regression_a(data, col_idx_dict)
    
    # investigate logistic regression
    X,y,data_filt = classification_models(data, col_idx_dict)


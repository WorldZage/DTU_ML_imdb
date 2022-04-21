# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from platform import system
from os import getcwd
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show, legend, ylim, imread
import sklearn.linear_model as lm

# internal scipts
from toolbox_02450 import windows_graphviz_call
import summary_statistics as su
import data_generator as dg
import dataloading_part2 as dl2
from constants import *


from cross_validation import cross_validate_lambda, cross_validate_feature
from classification import compare_models



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


def regression_a(data, col_idx_dict):

    cor_mat = np.corrcoef(data.T).round(3)
    print(f"{col_idx_dict}")
    # print(cor_mat)

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
    # it's difficult to imagine a linear relation between starting year, and movie rate.
    # Some periods of time might have higher rated movies than others.
    # Overall we might see an average tendency to decrease or increase in rating, but it won't help us with predictions.
    # So we remove it from our dataset.
    startYear_idx = col_idx_dict[startYear_name]
    X = np.delete(X, startYear_idx, 1)
    del feature_names[feature_names.index(startYear_name)]

    # Some features are spread across a large scale values.
    # For example, some movies have several thousand likes, while others only have tens.
    # to make them comparable, we use logarithmic transformation on popularity features
    # (but we will have to remove rows that contain 0 in any of these attributes)
    for name in [numVotes_name, movie_popularity_name, cast_popularity_name, nUser_reviews_name, nCritic_reviews_name]:
        idx = feature_names.index(name)


        # X[idx] = np.log(X[idx])

    # Get the dimensions of the data
    N, M = X.shape
    print(f"{N = }, {M = }")
    # Transform the data to be centered:
    X_center = X - np.ones((N, 1)) * X.mean(axis=0)
    # Transform the data to be normalized
    X_norm = X_center * (1 / np.std(X_center, axis=0))
    # print(f"{X_norm[0:3, :]}")


    lambdas = np.array([10 ** -5, 10 ** -3, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4])
    cross_validate_lambda(X_norm, y, 10, feature_names, lambdas)
    cross_validate_feature(X_norm, y, 10, feature_names)
    

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
    
    np.random.shuffle(data_lowrated)
    
    limit_sample = count_highrated
    print(limit_sample)
    data_lowrated_sample = np.empty((limit_sample,M))
    for i in range(limit_sample):
        data_lowrated_sample[i,:] = data_lowrated[i, :]
    
    data_filt = np.concatenate((data_highrated, data_lowrated_sample), axis=0)
    
    np.random.shuffle(data_filt)

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
        if y[i] > rate_class_limit:
            y_binary[i]=0
        else:
            y_binary[i]=1
        
    compare_models(X, y_binary)
    
    
    return X, y_binary, data_filt
    

if __name__ == '__main__':

    """ Movie data part:"""
    #write_filtered_and_movie_metadata_to_file()
    
    df_movies = pd.read_csv("collected_movie_data.csv", sep="\t", dtype=str)

    # convert to array
    data, col_idx_dict, col_idx_arr = dl2.data_loading(df_movies, "df_movies_and_extra")
    data = np.array(data, dtype=float)
    
    #normalize
    data_norm = data * (1/np.std(data,axis=0))
    cov_mat = np.cov(data_norm.T).round(3)
    #print(f"cov.mat: {cov_mat}")
    
    # investigate linear regression
    #regression_a(data, col_idx_dict)
    
    # investigate logistic regression
    X,y,data_filt = classification_models(data, col_idx_dict)


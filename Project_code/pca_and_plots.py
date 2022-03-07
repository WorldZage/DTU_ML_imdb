# this script is used for doing the PCA and displaying it using plots
import pandas as pd
import numpy as np
from constants import runtime_name, startYear_name, endYear_name, durationYears_name, nEpisodes_name, averageRating_name, \
    numVotes_name, genres_name
import matplotlib.pyplot as plt


def data_loading(df: pd.DataFrame,y_attr_name):
    col_names = list(df.columns)
    # extract our y data, the averageRating:
    y_col_idx = col_names.index(y_attr_name)
    y = np.asarray(df.values[:, y_col_idx],dtype=float)

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
            col_data = np.asarray(df.values[:, col_names.index(attr)], dtype=float)
            X_cols.append(col_data)

    # encoding the genres:
    """if genres_name in X_col_names:
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
        """
    X = np.vstack(X_cols)
    return y, X, attr_to_X_col_idx


def visualize(y, X, col_idx_dict):
    startYear_col = col_idx_dict[startYear_name]
    plt.plot(X[startYear_col], y, 'o', alpha=.1)
    plt.show()


def PCA(X):
    N = len(X[:, 0])
    print(N)
    # Subtract mean value from data
    Y = X - np.ones((N, 1)) * X.mean(axis=0)
    print((1/np.std(Y,axis=0)))
    Y = Y * (1/np.std(Y,axis=0))
    # PCA by computing SVD of Y
    U, S, V = np.linalg.svd(Y, full_matrices=False)

    # Compute variance explained by principal components
    rho = (S * S) / (S * S).sum()

    threshold = 0.9

    # Plot variance explained
    plt.figure()
    plt.plot(range(1, len(rho) + 1), rho, 'x-')
    plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-')
    plt.plot([1, len(rho)], [threshold, threshold], 'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual', 'Cumulative', 'Threshold'])
    plt.grid()
    plt.show()

def PCA_bar_plot(X, attributeNames):
    N, M = X.shape
    Y = X - np.ones((N, 1)) * X.mean(0)
    Y = Y * (1 / np.std(Y, axis=0))

    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    V = Vh.T
    print(attributeNames)
    pcs = [0, 1, 2, 3, 5, 6]
    legendStrs = ['PC' + str(e + 1) for e in pcs]
    c = ['r', 'g', 'b']
    bw = .1
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, attributeNames)
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('IMDb: PCA Component Coefficients')
    plt.show()
# this script is used for doing the PCA and displaying it using plots
import pandas as pd
import numpy as np
from constants import runtime_name, startYear_name, endYear_name, durationYears_name, nEpisodes_name, averageRating_name, \
    numVotes_name, genres_name
import matplotlib.pyplot as plt


def visualize(X_axis_data, Y_axis_data):
    plt.title("startyear vs avg. rating")
    plt.xlabel("start year")
    plt.ylabel("avg. rating, from 0-10")
    plt.plot(X_axis_data, Y_axis_data, 'o', alpha=.1)
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
    pcs = [0, 1, 2, 3]
    legendStrs = ['PC' + str(e + 1) for e in pcs]
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

def norm_plots(num_data, col_names):
    for i, col in enumerate(num_data):
        plt.hist(col, 20)
        plt.ylabel("occurrences")
        plt.xlabel(col_names[i])
        plt.title(col_names[i])
        plt.grid(axis="both")
        plt.show()


def project_plot(y_data, num_data):
    X = num_data
    N, M = X.shape
    # Subtract mean value from data
    Y = X - np.ones((N, 1)) * X.mean(0)
    Y = Y * (1 / np.std(Y, axis=0))

    # PCA by computing SVD of Y
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T

    # Project the centered data onto principal component space
    Z = Y @ V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title('IMDb data: PCA')
    # Z = array(Z)
    classNames = []
    print(y_data)
    for c in range(0,10,2):
        classNames.append(f"{c} < rating < {c + 2}")
        # select indices belonging to class c:
        class_mask = np.logical_and((c) < y_data, y_data < c + 2)
        plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.2)
    plt.grid(axis="both")
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i + 1))
    plt.ylabel('PC{0}'.format(j + 1))

    # Output result to screen
    plt.show()

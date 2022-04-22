# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:19:21 2022

@author: marle
"""

import pandas as pd
import numpy as np
from constants import runtime_name, startYear_name, endYear_name, durationYears_name, nEpisodes_name, averageRating_name, \
    numVotes_name, genres_name
import matplotlib.pyplot as plt

from sklearn import tree
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.pylab import imread, figure, plot, xlabel, ylabel, legend, ylim, show
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm


# def decision_trees(y, X, attributeNames):
#     print("applying ex 5.1 \n")
    
#     y_class = []
#     for i in y:
#         if i>8:
#             y_class.append(1)
#         else:
#             y_class.append(0)

#     # Fit regression tree classifier, Gini split criterion, no pruning
#     criterion = 'gini'
#     dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=50)
#     dtc = dtc.fit(X.T, y_class)
    
#     fname = 'tree_' + criterion
#     # Export tree graph .gvz file to parse to graphviz
#     out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)
    
#     # N.B.: you have to update the path_to_graphviz to reflect the position you 
#     # unzipped the software in!
#     windows_graphviz_call(fname=fname,
#                           cur_dir=getcwd(),
#                           path_to_graphviz=r'C:\Program Files\Graphviz')
#     plt.figure(figsize=(12, 12))
#     plt.imshow(imread(fname + '.png'))
#     plt.box('off');
#     plt.axis('off')
#     plt.show()
    

# def linear_regression(y, X):
#     print("applying ex 5.2 \n")
    
#     # Class names
#     #classNames = ['HighRated', 'LowRated']
    

#     # Fit ordinary least squares regression model
#     model = lm.LinearRegression()
#     model.fit(X,y)

#     # Predict average ratings
#     y_est = model.predict(X)
#     residual = y_est-y

#     # Display scatter plot
#     figure()
#     subplot(2,1,1)
#     plot(y, y_est, '.')
#     xlabel('Average Rating (true)'); ylabel('Average Rating (estimated)');
#     subplot(2,1,2)
#     hist(residual,40)

#     show()
    
#     return y, X


def logistic_regression(y,X, rate_class_limit):
    model = lm.LogisticRegression()
    model = model.fit(X,y)

    # Classify wine as White/Red (0/1) and assess probabilities
    y_est = model.predict(X)
    y_est_white_prob = model.predict_proba(X)[:, 0] 

    # # Define a new data object (new type of wine), as in exercise 5.1.7
    # x = np.array([6.9, 1.09, .06, 2.1, .0061, 12, 31, .99, 3.5, .44, 12]).reshape(1,-1)
    # # Evaluate the probability of x being a white wine (class=0) 
    # x_class = model.predict_proba(x)[0,0]

    # # Evaluate classifier's misclassification rate over entire training data
    # misclass_rate = np.sum(y_est != y) / float(len(y_est))

    # # Display classification results
    # print('\nProbability of given sample being a white wine: {0:.4f}'.format(x_class))
    # print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

    f = figure();
    class0_ids = np.nonzero(y==0)[0].tolist()
    plot(class0_ids, y_est_white_prob[class0_ids], '.y')
    class1_ids = np.nonzero(y==1)[0].tolist()
    plot(class1_ids, y_est_white_prob[class1_ids], '.r')
    xlabel('Data object (movie sample)'); ylabel('Predicted prob. of class >' + rate_class_limit + ' rated');
    legend(['Low Rated <' + rate_class_limit, 'High Rated >'+ rate_class_limit])
    ylim(-0.01,1.5)

    show()

    
    
    

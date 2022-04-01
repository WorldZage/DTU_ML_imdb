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
from matplotlib.image import imread
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm


def decision_trees(y, X, attributeNames):
    print("applying ex 5.1 \n")
    
    y_class = []
    for i in y:
        if i>8:
            y_class.append(1)
        else:
            y_class.append(0)

    # Fit regression tree classifier, Gini split criterion, no pruning
    criterion = 'gini'
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=50)
    dtc = dtc.fit(X.T, y_class)
    
    fname = 'tree_' + criterion
    # Export tree graph .gvz file to parse to graphviz
    out = tree.export_graphviz(dtc, out_file=fname + '.gvz', feature_names=attributeNames)
    
    # N.B.: you have to update the path_to_graphviz to reflect the position you 
    # unzipped the software in!
    windows_graphviz_call(fname=fname,
                          cur_dir=getcwd(),
                          path_to_graphviz=r'C:\Program Files\Graphviz')
    plt.figure(figsize=(12, 12))
    plt.imshow(imread(fname + '.png'))
    plt.box('off');
    plt.axis('off')
    plt.show()
    

def linear_regression(y, X):
    print("applying ex 5.2 \n")
    
    # Class names
    #classNames = ['HighRated', 'LowRated']
    

    # Fit ordinary least squares regression model
    model = lm.LinearRegression()
    model.fit(X,y)

    # Predict average ratings
    y_est = model.predict(X)
    residual = y_est-y

    # Display scatter plot
    figure()
    subplot(2,1,1)
    plot(y, y_est, '.')
    xlabel('Average Rating (true)'); ylabel('Average Rating (estimated)');
    subplot(2,1,2)
    hist(residual,40)

    show()
    
    return y, X
    
    

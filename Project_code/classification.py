# -*- coding: utf-8 -*-
"""
Compare Logistic Regression, Decision trees and Baseline
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm
from sklearn import  model_selection, tree
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from platform import system
from os import getcwd
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show, \
    legend, ylim, imread, boxplot, semilogx, loglog, title, grid

# internal scipts
from toolbox_02450 import rlr_validate
from toolbox_02450 import windows_graphviz_call
from toolbox_02450 import mcnemar
import summary_statistics as su
import data_generator as dg
import dataloading_part2 as dl2
from constants import *

def compare_models(X,y):
    N, M = X.shape
    
    font_size = 15
    plt.rcParams.update({'font.size': font_size})
    
    # k-fold crossvalidaton
    K = 10
    CV = model_selection.KFold(K, shuffle=True)    
    
    k = 0
    for train_index, test_index in CV.split(X,y):
        print("\n------------------------------------------------")
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
        
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    
        # Standardize the training and set set based on training set mean and std
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        ###################### INNER LOOP #####################################
        
        # K-fold crossvalidation INNER fold
        K_in = 10
        CV_in = model_selection.KFold(n_splits=K_in,shuffle=True)
        
        # init
        mu = np.empty((K_in, M-1))
        sigma = np.empty((K_in, M-1))
        
        # init LOGISTIC REGRESSION
        lambda_interval = np.logspace(-10, 3, 20)
        Error_train_logr = np.empty((len(lambda_interval),K_in))
        Error_test_logr = np.empty((len(lambda_interval),K_in))
        
        # init DECISION TREES
        # Tree complexity parameter - constraint on maximum depth
        tc = np.arange(2, 16, 1)
        Error_train_dt = np.empty((len(tc),K_in))
        Error_test_dt = np.empty((len(tc),K_in))
            
        k_in=0
        for train_index_in, test_index_in in CV_in.split(X_train):
            print('Computing CV inner fold: {0}/{1}..'.format(k_in+1,K_in))
            
            # extract training and test set for current CV fold
            X_train_in, y_train_in = X[train_index_in,:], y[train_index_in]
            X_test_in, y_test_in = X[test_index_in,:], y[test_index_in]
        
            mu = np.mean(X_train_in, 0)
            sigma = np.std(X_train_in, 0)
            
            X_train_in = (X_train_in - mu) / sigma
            X_test_in = (X_test_in - mu) / sigma
            
            ######################################
            # MODEL ##############################
            
            # LOGISTIC REGRESSION
            
            for lamb in range(0, len(lambda_interval)):
                mdl = LogisticRegression(penalty='l2', 
                                         C=1/(lambda_interval[lamb]))
                
                mdl.fit(X_train_in, y_train_in)
            
                y_est_train = mdl.predict(X_train_in).T
                y_est_test = mdl.predict(X_test_in).T

                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test_in) / float(len(y_est_test))
                misclass_rate_train = np.sum(y_est_train != y_train_in) / float(len(y_est_train))
                Error_test_logr[lamb,k_in], Error_train_logr[lamb,k_in] = misclass_rate_test, misclass_rate_train
            
            
            # DECISION TREES
            
            for i, t in enumerate(tc):
                # Fit decision tree classifier, Gini split criterion, different pruning levels
                dtc = tree.DecisionTreeClassifier(criterion='gini', 
                                                  max_depth=t,
                                                  splitter="best")
                dtc = dtc.fit(X_train_in,y_train_in.ravel())
                y_est_test = dtc.predict(X_test_in)
                y_est_train = dtc.predict(X_train_in)
                
                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test_in) / float(len(y_est_test))
                misclass_rate_train = np.sum(y_est_train != y_train_in) / float(len(y_est_train))
                Error_test_dt[i,k_in], Error_train_dt[i,k_in] = misclass_rate_test, misclass_rate_train


            k_in+=1
            
        ################## CLOSE INNER LOOP ###################################
        
        
        #####################################
        # PLOT, ERROR RATES #################
        
        # LOGISTIC REGRESSION
        error_fold = Error_test_logr.mean(1)
        opt_test_error = np.min(error_fold)
        opt_lambda_idx = np.argmin(error_fold)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        plt.figure(figsize=(8,8))
        plt.plot(np.log10(lambda_interval), Error_train_logr.mean(1)*100)
        plt.plot(np.log10(lambda_interval), Error_test_logr.mean(1)*100)
        plt.plot(np.log10(opt_lambda), opt_test_error*100, 'o')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.grid()
        plt.show()


        # DECISION TREES
        
        error_fold = Error_test_dt.mean(1)
        error_fold_lst = error_fold.tolist()
        opt_test_error = min(error_fold_lst)
        opt_model_complex = error_fold_lst.index(opt_test_error) +2
        
        figure()
        boxplot(Error_test_dt.T)
        xlabel('Model complexity (max tree depth)')
        ylabel('Test error across CV folds, K={0})'.format(K_in))
        
        figure()
        plot(tc, Error_train_dt.mean(1))
        plot(tc, Error_test_dt.mean(1))
        xlabel('Model complexity (max tree depth)')
        ylabel('Error (misclassification rate, CV K={0})'.format(K_in))
        legend(['Error_train','Error_test'])
        show()                
        
        
        #####################################
        # TEST ON OUTER SPLIT ###############
        
        print("\n FOLD MISCLASSIFICATION ERROR RATE:")
        
        
        # LOGISTIC REGRESSION
        
        mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
        
        mdl.fit(X_train, y_train)
        y_est_test_logr = mdl.predict(X_test).T
        
        test_error_rate_logr = np.sum(y_est_test_logr != y_test) / len(y_test)
        w_est = mdl.coef_[0]
        
        print("\nLogistic Regression:")
        print("Optimal Lambda:", opt_lambda)
        print("Logistic Regression Error:",test_error_rate_logr*100, "%")
        print("coefficients", w_est)
        
            # NO REGULARIZATION
            
        mdl_noreg = LogisticRegression(penalty='l2')
        
        mdl_noreg.fit(X_train, y_train)
        y_est_test_logr_noreg = mdl_noreg.predict(X_test).T
        
        test_error_rate_logr_noreg = np.sum(y_est_test_logr_noreg != y_test) / len(y_test)
        
        w_est_noreg = mdl.coef_[0]
        
        print("\nLogistic Regression No regularization:")
        print("Logistic Regression Error:",test_error_rate_logr_noreg*100, "%")
        print("coefficients", w_est_noreg)
        
        
        # DECISION TREES
        
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_model_complex)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test_dt = dtc.predict(X_test)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        test_error_rate_dt = np.sum(y_est_test_dt != y_test) / float(len(y_test))
        
        print("\nDecision Trees:")
        print("Optimal Model Complexity", opt_model_complex)
        print("Decision Trees Error: ",test_error_rate_dt*100, "%")
        
        
        
        # BASELINE
        
        y_est_test_bas = np.full(len(y_test), round(y_train.mean()) ) # ones
        # Evaluate misclassification rate over train/test data (in this CV fold)
        test_error_rate_bas = np.sum(y_est_test_bas != y_test) / float(len(y_est_test))
        
        # THEY ARE TRAINED IN THE TRAINING SETS AND EVALUTARED IN TEST SETS?
        print("\nBaseline:")
        print("Baseline Error: ",test_error_rate_bas*100, "%")
        
        
        
        #####################################
        # STATISTICAL EVALUATION ############
        # Compute the McNemar interval
        print("\n. . . . . . . . . . . . . . . . . . .")
        
        print("length y_test", len(y_test))

        # LOGISTICAL REGRESSION VS DECISION TREE
        print("\nLOGISTICAL REGRESSION VS DECISION TREE")
               
        alpha = 0.05
        [thetahat, CI, p] = mcnemar(y_test, y_est_test_logr, y_est_test_dt, alpha=alpha)

        print("theta = theta_A-theta_B point estimate", thetahat, 
              "\n CI: ", CI, 
              "\n p-value", p)
        
        # LOGISTICAL REGRESSION VS BASELINE
        print("\nLOGISTICAL REGRESSION VS BASELINE")
        
        alpha = 0.05
        [thetahat, CI, p] = mcnemar(y_test, y_est_test_logr, y_est_test_bas, alpha=alpha)

        print("theta = theta_A-theta_B point estimate", thetahat, 
              "\n CI: ", CI, 
              "\n p-value", p)

        # DECISION TREE VS BASELINE
        print("\nDECISION TREE VS BASELINE")
               
        alpha = 0.05
        [thetahat, CI, p] = mcnemar(y_test, y_est_test_dt, y_est_test_bas, alpha=alpha)

        print("theta = theta_A-theta_B point estimate", thetahat, 
              "\n CI: ", CI, 
              "\n p-value", p)
        
        k+=1

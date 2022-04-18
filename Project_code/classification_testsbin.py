# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 00:34:41 2022

@author: marle
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

def logistic_reg(X, y, rate_class_limit):
    # LOG REG MODEL ######
    
    model = lm.LogisticRegression()
    model = model.fit(X,y)

    # Classify wine as High/Low rated (0/1) and assess probabilities
    y_est = model.predict(X)
    y_est_high_rated_prob = model.predict_proba(X)[:, 0] 
    
    
    # plot
    f = figure();
    class0_ids = np.nonzero(y==0)[0].tolist()
    plot(class0_ids, y_est_high_rated_prob[class0_ids], '.y')
    
    class1_ids = np.nonzero(y==1)[0].tolist()
    plot(class1_ids, y_est_high_rated_prob[class1_ids], '.r')
    
    xlabel('Data object (movie sample)'); ylabel('Predicted prob. of class >' + str(rate_class_limit) + ' rated');
    legend(['High Rated >' + str(rate_class_limit), 'Low Rated <'+ str(rate_class_limit)])

    show()
    
def logistic_reg_lambda(X, y):
    
    N, M = X.shape
    
    font_size = 15
    plt.rcParams.update({'font.size': font_size})
    
    
    K = 5
    CV = model_selection.KFold(K, shuffle=True)    
    
    k = 0
    for train_index, test_index in CV.split(X,y):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
        
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    
        # Standardize the training and set set based on training set mean and std
        # not considering offset - WHY IS IT NEEDED?
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        ###################### INNER LOOP #####################################
        # Create crossvalidation partition for evaluation
        # using stratification and 97 pct. split between training and test 
        
        # K-fold crossvalidation INNER fold
        K_in = 5
        CV_in = model_selection.KFold(n_splits=K_in,shuffle=True)
        
        # init
        Error_train = np.empty((K_in,1))
        Error_test = np.empty((K_in,1))
        Error_train_rlr = np.empty((K_in,1))
        Error_test_rlr = np.empty((K_in,1))
        Error_train_nofeatures = np.empty((K_in,1))
        Error_test_nofeatures = np.empty((K_in,1))
        w_rlr = np.empty((M,K_in))
        mu = np.empty((K_in, M-1))
        sigma = np.empty((K_in, M-1))
        w_noreg = np.empty((M,K_in))
        
        # Fit regularized logistic regression model to training data to predict 
        # the type of wine
        lambda_interval = np.logspace(-8, 2, 70)
        # train_error_rate = np.zeros(len(lambda_interval))
        # test_error_rate = np.zeros(len(lambda_interval))
        # Initialize variable
        Error_train = np.empty((len(lambda_interval),K_in))
        Error_test = np.empty((len(lambda_interval),K_in))
        #coefficient_norm = np.zeros(len(lambda_interval))
            
        k_in=0
        for train_index_in, test_index_in in CV_in.split(X_train):
            print('Computing CV inner fold: {0}/{1}..'.format(k_in+1,K_in))
            
            # extract training and test set for current CV fold
            X_train_in, y_train_in = X[train_index_in,:], y[train_index_in]
            X_test_in, y_test_in = X[test_index_in,:], y[test_index_in]
        
            # X_train_in, X_test_in, y_train_in, y_test_in = train_test_split(
            #     X_train, y_train, test_size=.90, stratify=y_train)
            
            # Standardize the training and set set based on training set mean and std
            # not considering offset - WHY IS IT NEEDED?
            mu = np.mean(X_train_in, 0)
            sigma = np.std(X_train_in, 0)
            
            X_train_in = (X_train_in - mu) / sigma
            X_test_in = (X_test_in - mu) / sigma
            
            for lamb in range(0, len(lambda_interval)):
                mdl = LogisticRegression(penalty='l2', C=1/(lambda_interval[lamb]) )
                
                mdl.fit(X_train_in, y_train_in)
            
                y_est_train = mdl.predict(X_train_in).T
                y_est_test = mdl.predict(X_test_in).T


                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test_in) / float(len(y_est_test))
                misclass_rate_train = np.sum(y_est_train != y_train_in) / float(len(y_est_train))
                Error_test[lamb,k_in], Error_train[lamb,k_in] = misclass_rate_test, misclass_rate_train
                
                # train_error_rate[lamb] = np.sum(y_train_est != y_train_in) / len(y_train_in)
                # test_error_rate[lamb] = np.sum(y_test_est != y_test_in) / len(y_test_in)
            
                # weights - coefficient of the features in the decision function
                # REALLY DON'T UNDERSTAND THIS
                # w_est = mdl.coef_[0] 
                # coefficient_norm[lamb] = np.sqrt(np.sum(w_est**2))
                
            k_in+=1
            
            ##################### CLOSE INNER LOOP
        error_fold = Error_test.mean(1)
            
        opt_test_error = np.min(error_fold)
        opt_lambda_idx = np.argmin(error_fold)
        opt_lambda = lambda_interval[opt_lambda_idx]

        # min_error = np.min(test_error_rate)
        # opt_lambda_idx = np.argmin(test_error_rate)
        # opt_lambda = lambda_interval[opt_lambda_idx]
        
        # error + final parameter
       
        # error_fold = Error_test.mean(1)
        # print(error_fold)
        # error_fold_lst = error_fold.tolist()
        # opt_test_error = min(error_fold_lst)
        # opt_lambda = error_fold_lst.index(opt_test_error)
        
        print("\n Minimum Test Error", opt_test_error)
        print("Optimal Lambda", opt_lambda)
        
        # print("min_error", min_error)
        # print("opt_lambda_idx", opt_lambda_idx)
        # print("opt_lambda", opt_lambda)
        
        # print("Minimum test error: " + str(np.round(min_error*100,2)) + 
        #       ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
        
        #PLOT
        plt.figure(figsize=(8,8))
        plt.plot(np.log10(lambda_interval), Error_train.mean(1)*100)
        plt.plot(np.log10(lambda_interval), Error_test.mean(1)*100)
        plt.plot(np.log10(opt_lambda), opt_test_error*100, 'o')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.grid()
        plt.show()

        # TEST ON OUTER TEST SPLIT

        mdl = LogisticRegression(penalty='l2', C=1/opt_lambda)
        
        mdl.fit(X_train, y_train)
        y_test_est = mdl.predict(X_test).T
        
        test_error_rate_fold = np.sum(y_test_est != y_test) / len(y_test)
        print("misclass_rate_test",test_error_rate_fold, "\n")
    
        
    
        # BASELINE
        
        y_est_test_bas = np.full(len(y_train), round(y_test.mean()) ) # ones
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test_bas = np.sum(y_est_test_bas != y_test) / float(len(y_est_test))
        
        # THEY ARE TRAINED IN THE TRAINING SETS AND EVALUTARED IN TEST SETS?
        print("misclass_rate_test baseline",misclass_rate_test_bas, "\n")
            

        k+=1


def k_nearest(X, y):
    
    N, M = X.shape
    
    # Maximum number of neighbors
    L=20

    K=3
    CV = model_selection.KFold(K, shuffle=True)
    errors = np.zeros((K,L))
    i=0
    for train_index, test_index in CV.split(X, y):
        print('Crossvalidation fold: {0}/{1}'.format(i+1,K))    
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]

        C=2
        # Plot the training data points (color-coded) and test data points.
        figure()
        class_mask = (y_train==0)
        plot(X_train[class_mask,0], X_train[class_mask,1], '.g', alpha=0.5)
        figure()
        class_mask = (y_train==1)
        plot(X_train[class_mask,0], X_train[class_mask,1], '.r', alpha=0.5)

    #     # Fit classifier and classify the test points (consider 1 to L neighbors)
    #     for l in range(1,L+1):
    #         knclassifier = KNeighborsClassifier(n_neighbors=l);
    #         knclassifier.fit(X_train, y_train);
    #         y_est = knclassifier.predict(X_test);
    #         #if condiction applies, it will give 1
    #         errors[i,l-1] = np.sum(y_est[0]!=y_test[0])
    #         print("y_est[0]",y_est[0])
    #         print("y_test[0]",y_test[0])
    #     print("errors[i,]",errors[i,:])
    #     input()
    #         #i = current fold
    #         #l = num neighbors
        i+=1
    #print(errors)
    ## print("min error",min_error, "at", np.where(errors == min_error))
    ## Plot the classification error rate
    #figure()
    #plot(100*sum(errors,0)/N)
    #xlabel('Number of neighbors')
    #ylabel('Classification error rate (%)')
    #show()
    
def decision_tree(X,y):
    
    # K-fold crossvalidation
    K = 5
    CV = model_selection.KFold(n_splits=K,shuffle=True)
    
    
    k=0
    for train_index, test_index in CV.split(X):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))
    
        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
        
        # Tree complexity parameter - constraint on maximum depth
        tc = np.arange(2, 16, 1)
    
        # K-fold crossvalidation INNER fold
        K_in = 5
        CV_in = model_selection.KFold(n_splits=K_in,shuffle=True)

        # Initialize variable
        Error_train = np.empty((len(tc),K_in))
        Error_test = np.empty((len(tc),K_in))
    
        k_in=0
        for train_index_in, test_index_in in CV_in.split(X_train):
            print('Computing CV inner fold: {0}/{1}..'.format(k_in+1,K_in))
            
            # extract training and test set for current CV fold
            X_train_in, y_train_in = X[train_index_in,:], y[train_index_in]
            X_test_in, y_test_in = X[test_index_in,:], y[test_index_in]
        
            for i, t in enumerate(tc):
                # Fit decision tree classifier, Gini split criterion, different pruning levels
                dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
                dtc = dtc.fit(X_train_in,y_train_in.ravel())
                y_est_test = dtc.predict(X_test_in)
                y_est_train = dtc.predict(X_train_in)
                # Evaluate misclassification rate over train/test data (in this CV fold)
                misclass_rate_test = np.sum(y_est_test != y_test_in) / float(len(y_est_test))
                misclass_rate_train = np.sum(y_est_train != y_train_in) / float(len(y_est_train))
                Error_test[i,k_in], Error_train[i,k_in] = misclass_rate_test, misclass_rate_train
            k_in+=1
            
        figure()
        boxplot(Error_test.T)
        xlabel('Model complexity (max tree depth)')
        ylabel('Test error across CV folds, K={0})'.format(K_in))
        
        figure()
        plot(tc, Error_train.mean(1))
        plot(tc, Error_test.mean(1))
        xlabel('Model complexity (max tree depth)')
        ylabel('Error (misclassification rate, CV K={0})'.format(K_in))
        legend(['Error_train','Error_test'])
            
        show()
        
        # error + final parameter
        
        error_fold = Error_test.mean(1)
        error_fold_lst = error_fold.tolist()
        opt_test_error = min(error_fold_lst)
        opt_model_complex = error_fold_lst.index(opt_test_error)
        
        print("\n Minimum Test Error", opt_test_error)
        print("Optimal Model Complexity", opt_model_complex+2)
        
        # TEST ON OUTER TEST SPLIT
        
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=opt_model_complex+2)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        print("misclass_rate_test",misclass_rate_test, "\n")
        
        k+=1
    

        
        
def reg_param_holdout(X,y):
    
    
    N, M = X.shape
    
    font_size = 15
    plt.rcParams.update({'font.size': font_size})
    
    
    K = 5
    CV = model_selection.KFold(K, shuffle=True)    
    # init
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K))
    
    k = 0
    for train_index, test_index in CV.split(X,y):
        print(k)
        
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    
        ###################### INNER LOOP #####################################
        # Create crossvalidation partition for evaluation
        # using stratification and 97 pct. split between training and test 
        X_train_in, X_test_in, y_train_in, y_test_in = train_test_split(
            X_train, y_train, test_size=.90, stratify=y_train)
        
        # Standardize the training and set set based on training set mean and std
        # not considering offset - WHY IS IT NEEDED?
        mu = np.mean(X_train_in, 0)
        sigma = np.std(X_train_in, 0)
        
        X_train_in = (X_train_in - mu) / sigma
        X_test_in = (X_test_in - mu) / sigma
        
        # Fit regularized logistic regression model to training data to predict 
        # the type of wine
        lambda_interval = np.logspace(-8, 2, 70)
        train_error_rate = np.zeros(len(lambda_interval))
        test_error_rate = np.zeros(len(lambda_interval))
        #coefficient_norm = np.zeros(len(lambda_interval))
        
        for k_inner in range(0, len(lambda_interval)):
            mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k_inner] )
            
            mdl.fit(X_train_in, y_train_in)
        
            y_train_est = mdl.predict(X_train_in).T
            print(y_train_est)
            y_test_est = mdl.predict(X_test_in).T
            print(y_test_est)
            
            train_error_rate[k_inner] = np.sum(y_train_est != y_train_in) / len(y_train_in)
            test_error_rate[k_inner] = np.sum(y_test_est != y_test_in) / len(y_test_in)
        
            # weights - coefficient of the features in the decision function
            # REALLY DON'T UNDERSTAND THIS
            # w_est = mdl.coef_[0] 
            # coefficient_norm[k_inner] = np.sqrt(np.sum(w_est**2))
        
        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        
        print("min_error", min_error)
        print("opt_lambda_idx", opt_lambda_idx)
        print("opt_lambda", opt_lambda)
        
        print("Minimum test error: " + str(np.round(min_error*100,2)) + 
              ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
        
        #PLOT
        plt.figure(figsize=(8,8))
        plt.plot(np.log10(lambda_interval), train_error_rate*100)
        plt.plot(np.log10(lambda_interval), test_error_rate*100)
        plt.plot(np.log10(opt_lambda), min_error*100, 'o')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.grid()
        plt.show()
        
        # CLOSE INNER LOOP
        #Standardize the training and set set based on training set mean and std
        #not considering offset - WHY IS IT NEEDED?
        mu = np.mean(X_train, 0)
        sigma = np.std(X_train, 0)
        
        X_train = (X_train - mu) / sigma
        X_test = (X_test - mu) / sigma
        
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        #Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
        #Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        #Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
        #Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        # Display results
        print('Linear regression without feature selection:')
        print('- Training error: {0}',Error_train[k])
        print('- Test error:     {0}',Error_test[k])
        print('Regularized linear regression:')
        print('- Training error: {0}', Error_train_rlr[k])
        print('- Test error:     {0}', Error_test_rlr[k])
        k+=1
"""

LOOKS LIKE THE TAMPON FUNCTION::::

    plt.figure(figsize=(8,8))
    plt.semilogx(lambda_interval, coefficient_norm,'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.show()
    

        # Xty = X_train.T @ y_train
        # XtX = X_train.T @ X_train
        # # Compute mean squared error without using the input data at all
        # Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        # Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    
        # # Estimate weights for the optimal value of lambda, on entire training set
        # lambdaI = opt_lambda * np.eye(M)
        # lambdaI[0,0] = 0 # Do no regularize the bias term
        # w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # # Compute mean squared error with regularization with optimal lambda
        # Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        # Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
        # # Estimate weights for unregularized linear regression, on entire training set
        # w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # # Compute mean squared error without regularization
        # Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        # Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        

        # # Display results
        # print('Linear regression without feature selection:')
        # print('- Training error: {0}',Error_train[k])
        # print('- Test error:     {0}',Error_test[k])
        # print('Regularized linear regression:')
        # print('- Training error: {0}', Error_train_rlr[k])
        # print('- Test error:     {0}', Error_test_rlr[k])

"""

def reg_param_kfold(X, y):
    
    N, M = X.shape
    
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    #CV = model_selection.KFold(K, shuffle=False)
    
    # Fit regularized logistic regression model to training data to predict 
    # the imdb score
    lambda_interval = np.power(10.,range(-5,8))#np.logspace(-8, 2, 500)
    train_error_rate = np.zeros(len(lambda_interval))
    test_error_rate = np.zeros(len(lambda_interval))
    coefficient_norm = np.zeros(len(lambda_interval))
    
    k=0
    for train_index, test_index in CV.split(X,y):
        #5 voltinhas

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
    
        # Logistic
        mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
        
        mdl.fit(X_train, y_train)
    
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
        test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)
    
        # weights - coefficient of the features in the decision function
        w_est = mdl.coef_[0] 
        coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    

        
        #To inspect the used indices, use these print statements
        print('Cross validation fold {0}/{1}:'.format(k+1,K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}\n'.format(test_index))
    
        k+=1
    
    min_error = np.min(test_error_rate)
    opt_lambda_idx = np.argmin(test_error_rate)
    opt_lambda = lambda_interval[opt_lambda_idx]
    
    print("Minimum test error: " + str(np.round(min_error*100,2)) + 
          ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
    
    plt.figure(figsize=(8,8))
    plt.plot(np.log10(lambda_interval), train_error_rate*100)
    plt.plot(np.log10(lambda_interval), test_error_rate*100)
    plt.plot(np.log10(opt_lambda), min_error*100, 'o')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
    plt.grid()
    plt.show()    
    
    plt.figure(figsize=(8,8))
    plt.semilogx(lambda_interval, coefficient_norm,'k')
    plt.ylabel('L2 Norm')
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.title('Parameter vector L2 norm')
    plt.grid()
    plt.show() 
    
    show()

    
    print('Ran Exercise 8.1.1')
        

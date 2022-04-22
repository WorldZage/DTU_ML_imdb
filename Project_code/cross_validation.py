import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, xlabel, ylabel, clim, title, show, semilogx, grid, loglog, legend
import scipy.stats as st
from sklearn import model_selection
import sklearn.linear_model as lm
from sklearn.model_selection import ShuffleSplit
from toolbox_02450 import feature_selector_lr, bmplot, rlr_validate, train_neural_net, draw_neural_net
import torch
import os, sys


def cross_validate_feature(X, y, K, attributeNames):
    N, M = X.shape

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    # Initialize variables
    Features = np.zeros((M, K))
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_fs = np.empty((K, 1))
    Error_test_fs = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))

    k = 0
    for train_index, test_index in CV.split(X):

        # extract training and test set for current CV fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        internal_cross_validation = 10

        # Compute squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum() / y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum() / y_test.shape[0]

        # Compute squared error with all features selected (no feature selection)
        m = lm.LinearRegression(fit_intercept=True).fit(X_train, y_train)
        Error_train[k] = np.square(y_train - m.predict(X_train)).sum() / y_train.shape[0]
        Error_test[k] = np.square(y_test - m.predict(X_test)).sum() / y_test.shape[0]

        # Compute squared error with feature subset selection
        textout = ''
        selected_features, features_record, loss_record = feature_selector_lr(X_train, y_train,
                                                                              internal_cross_validation,
                                                                              display=textout)

        Features[selected_features, k] = 1
        # .. alternatively you could use module sklearn.feature_selection
        if len(selected_features) == 0:
            print('No features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
        else:
            m = lm.LinearRegression(fit_intercept=True).fit(X_train[:, selected_features], y_train)
            Error_train_fs[k] = np.square(y_train - m.predict(X_train[:, selected_features])).sum() / y_train.shape[0]
            Error_test_fs[k] = np.square(y_test - m.predict(X_test[:, selected_features])).sum() / y_test.shape[0]

            figure(k)
            subplot(1, 2, 1)
            plot(range(1, len(loss_record)), loss_record[1:])
            xlabel('Iteration')
            ylabel('Squared error (crossvalidation)')

            subplot(1, 3, 3)
            bmplot(attributeNames, range(1, features_record.shape[1]), -features_record[:, 1:])
            clim(-1.5, 0)
            xlabel('Iteration')

        print('Cross validation fold {0}/{1}'.format(k + 1, K))
        print('Train indices: {0}'.format(train_index))
        print('Test indices: {0}'.format(test_index))
        print('Features no: {0}\n'.format(selected_features.size))

        k += 1

    # Display results
    print('\n')
    print('Linear regression without feature selection:\n')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
    print('Linear regression with feature selection:\n')
    print('- Training error: {0}'.format(Error_train_fs.mean()))
    print('- Test error:     {0}'.format(Error_test_fs.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train_fs.sum()) / Error_train_nofeatures.sum()))
    print(
        '- R^2 test:     {0}'.format((Error_test_nofeatures.sum() - Error_test_fs.sum()) / Error_test_nofeatures.sum()))

    figure(k)
    subplot(1, 3, 2)
    bmplot(attributeNames, range(1, Features.shape[1] + 1), -Features)
    clim(-1.5, 0)
    xlabel('Crossvalidation fold')
    ylabel('Attribute')

    # Inspect selected feature coefficients effect on the entire dataset and
    # plot the fitted model residual error as function of each attribute to
    # inspect for systematic structure in the residual

    f = 2  # cross-validation fold to inspect
    ff = Features[:, f - 1].nonzero()[0]
    if len(ff) == 0:
        print('\nNo features were selected, i.e. the data (X) in the fold cannot describe the outcomes (y).')
    else:
        m = lm.LinearRegression(fit_intercept=True).fit(X[:, ff], y)

        y_est = m.predict(X[:, ff])
        residual = y - y_est

        figure(k + 1, figsize=(12, 6))
        title('Residual error vs. Attributes for features selected in cross-validation fold {0}'.format(f))
        for i in range(0, len(ff)):
            subplot(2, np.ceil(len(ff) / 2), i + 1)
            plot(X[:, ff[i]], residual, '.')
            xlabel(attributeNames[ff[i]])
            ylabel('residual error')

    show()


def cross_validate_lambda(X, y, K, attributeNames, lambdas=np.power(10., range(-5, 9))):
    N, M = X.shape

    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    attributeNames = [u'Offset'] + attributeNames
    M = M + 1
    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(K, shuffle=True)
    # CV = model_selection.KFold(K, shuffle=False)

    # Initialize variables
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    opt_lambda_lst = []

    k = 0
    for train_index, test_index in CV.split(X, y):

        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10

        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train,
                                                                                                          y_train,
                                                                                                          lambdas,
                                                                                                          internal_cross_validation)
        opt_lambda_lst.append(opt_lambda)
        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
        Error_test[k] = np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
        # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
        # m = lm.LinearRegression().fit(X_train, y_train)
        # Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        # Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

        # Display the results for the last cross-validation fold
        if k == K - 1:
            figure(k, figsize=(12, 8))
            subplot(1, 2, 1)
            semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()
            # You can choose to display the legend, but it's omitted for a cleaner
            # plot, since there are many attributes
            legend(attributeNames[1:], loc='best')

            subplot(1, 2, 2)
            title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
            loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            legend(['Train error', 'Validation error'])
            grid()

        # To inspect the used indices, use these print statements
        # print('Cross validation fold {0}/{1}:'.format(k+1,K))
        # print('Train indices: {0}'.format(train_index))
        # print('Test indices: {0}\n'.format(test_index))

        k += 1

    show()
    # Display results
    print('Linear regression without feature selection:')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train.sum()) / Error_train_nofeatures.sum()))
    print(
        '- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum() - Error_test.sum()) / Error_test_nofeatures.sum()))
    print('Regularized linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format(
        (Error_train_nofeatures.sum() - Error_train_rlr.sum()) / Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format(
        (Error_test_nofeatures.sum() - Error_test_rlr.sum()) / Error_test_nofeatures.sum()))

    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m, -1], 2)))

    # My addition:
    print('- Avatar actual imdb: {} vs estimated {}'.format(y[0], X[0, :] @ w_rlr[:, -1]))
    print('List of optimal lambdas: {}'.format(opt_lambda_lst))


def cross_validate_ann(X, y, K, attributeNames):
    N, M = X.shape
    CV = model_selection.KFold(K, shuffle=True)
    y_reshape = y.reshape((N,
                           1))  # Reshape the y data to avoid shape issues. basically just make it into a singled-column matrix ( i think).

    # Parameters for neural network
    n_hidden_units = 2  # number of hidden units
    n_replicates = 3  # number of networks trained in each k-fold
    max_iter = 10000

    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = []  # make a list for storing generalization error in each loop

    best_test_err = np.inf
    for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
        print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y_reshape[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y_reshape[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)

        print('\n\tBest loss: {}\n'.format(final_loss))

        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors.append(mse)  # store error rate for current CV fold

        # MODIFIED:
        # 'Remember' the best CV fold.
        if mse < best_test_err:
            best_test_err = mse
            best_cv_data = (y_test, y_test_est)

        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k + 1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')

    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, K + 1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')

    print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0, 2]]
    biases = [net[i].bias.data.numpy() for i in [0, 2]]
    tf = [str(net[i]) for i in [1, 2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

    # Print the average classification error rate
    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10, 10))
    # MODIFIED:
    y_true = best_cv_data[0].data.numpy()  # y_test.data.numpy()
    y_est = best_cv_data[1].data.numpy()  # y_test_est.data.numpy();
    axis_range = [np.min([y_est, y_true]) - 1, np.max([y_est, y_true]) + 1]
    plt.plot(axis_range, axis_range, 'k--')
    plt.plot(y_true, y_est, 'ob', alpha=.25)
    plt.legend(['Perfect estimation', 'Model estimations'])
    plt.title('Rating: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range)
    plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()

    plt.show()


# Context Manager for 'muting' prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def optimal_hidden_unit_ann(X: np.ndarray, y: np.ndarray, hidden_unit_options, cvf=10):
    N, M = X.shape
    CV = model_selection.KFold(cvf, shuffle=True)

    # Normalize data
    # X = stats.zscore(X);

    # Reshape the y data to avoid shape issues. basically just make it into a singled-column matrix
    y_reshape = y.reshape((y.shape[0], 1))
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000

    """number of hidden units options"""
    n_hu_options = len(hidden_unit_options)

    # train_error = np.empty((cvf, n_hu_options))
    test_error = np.zeros((cvf, n_hu_options))

    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    for (k, (train_index, test_index)) in enumerate(CV.split(X, y_reshape)):
        print('\nCrossvalidation ANN fold: {0}/{1}'.format(k + 1, cvf))

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index])
        y_train = torch.Tensor(y_reshape[train_index])
        X_test = torch.Tensor(X[test_index])
        y_test = torch.Tensor(y_reshape[test_index])

        for n, current_hu in enumerate(hidden_unit_options):
            # Define the model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(M, current_hu),  # M features to n_hidden_units
                torch.nn.Tanh(),  # 1st transfer function,
                torch.nn.Linear(current_hu, 1),  # n_hidden_units to 1 output neuron
                # no final transfer function, i.e. "linear output"
            )

            # print('Training model of type:\n\n{}\n'.format(str(model())))

            # Train the net on training data

            with HiddenPrints():
                current_net, _, _ = train_neural_net(model,
                                                     loss_fn=loss_fn,
                                                     X=X_train,
                                                     y=y_train,
                                                     n_replicates=n_replicates,
                                                     max_iter=max_iter)

            # print('\n\tBest loss: {}\n'.format(final_loss))

            # Determine estimated class labels for test set
            y_test_est = current_net(X_test)

            # Determine errors and errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
            test_error[k, n] = mse
            """if mse < best_test_err:
                best_test_err = mse
                best_net = net
                opt_hidden_units = n_hidden_units
            """

    # print(f"{np.mean(test_error, axis=0) = }")
    opt_mse_err = np.min(np.mean(test_error, axis=0))
    # print(f"{np.argmin(np.mean(test_error, axis=0)) = }")
    opt_hidden_units = hidden_unit_options[np.argmin(np.mean(test_error, axis=0))]

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final transfer function, i.e. "linear output"
    )

    return opt_hidden_units, opt_mse_err


def sub_cross_validate_rlr(X, y, lambdas, partition, K=10):
    X_rlr = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    N, M_rlr = X_rlr.shape
    train_index, test_index = partition

    # extract training and test set for current CV fold for RLR
    X_rlr_train = X_rlr[train_index]
    y_train = y[train_index]
    X_rlr_test = X_rlr[test_index]
    y_test = y[test_index]

    # Standardize the train & test sets:
    mu = np.mean(X_rlr_train[:, 1:], 0)
    sigma = np.std(X_rlr_train[:, 1:], 0)

    X_rlr_train[:, 1:] = (X_rlr_train[:, 1:] - mu) / sigma
    X_rlr_test[:, 1:] = (X_rlr_test[:, 1:] - mu) / sigma

    Xty = X_rlr_train.T @ y_train
    XtX = X_rlr_train.T @ X_rlr_train

    # Regularized Linear Regression (RLR)
    _, opt_lambda, _, _, _ = rlr_validate(X_rlr_train,
                                          y_train,
                                          lambdas,
                                          K)
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M_rlr)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    rlr_y_test_est = X_rlr_test @ w_rlr
    return opt_lambda, rlr_y_test_est


def sub_cross_validate_ann(X, y, hidden_unit_options, partition, K=10, max_iter=1000):
    M_ann = X.shape[1]
    # reshape y vector for ANN
    y_ann = y.reshape((y.shape[0], 1))

    train_index, test_index = partition

    # Artificial Neural Network (ANN)
    # extract training and test set for current CV fold for RLR
    X_ann_train = X[train_index, :]
    y_train = y[train_index]
    X_ann_test = X[test_index, :]
    y_test = y[test_index]
    opt_hu, _ = optimal_hidden_unit_ann(X_ann_train, y_train, hidden_unit_options, cvf=K)  #

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M_ann, opt_hu),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_hu, 1),  # n_hidden_units to 1 output neuron
        # no final transfer function, i.e. "linear output"
    )

    # Doing ANN with the number of hidden units specified:
    X_train_tensor = torch.Tensor(X_ann_train)
    y_train_tensor = torch.Tensor(y.reshape(y.shape[0], 1)[train_index])
    X_test_tensor = torch.Tensor(X_ann_test)
    loss_fn = torch.nn.MSELoss()
    n_replicates = 1

    with HiddenPrints():
        opt_net, _, _ = train_neural_net(model,
                                         loss_fn,
                                         X=X_train_tensor,
                                         y=y_train_tensor,
                                         n_replicates=n_replicates,
                                         max_iter=max_iter)
    ann_y_test_est = opt_net(X_test_tensor).data.numpy().reshape(y_test.shape)  # .T
    return opt_hu, ann_y_test_est


def cross_validate_model_comparison(X, y, lambdas, hidden_unit_options, K=10):
    # Add offset attribute for RLR

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(n_splits=K, shuffle=True)

    baseline_test_errors = np.empty((K, 1))
    opt_lambda_lst, rlr_test_errors = [0] * K, [0] * K
    opt_n_hu, ann_test_errors = [0] * K, [0] * K

    """
    mu = np.empty((K, M_rlr - 1))
    sigma = np.empty((K, M_rlr - 1))
    """

    for (k, (train_index, test_index)) in enumerate(CV.split(X)):
        print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
        y_test = y[test_index]

        partition = (train_index, test_index)

        opt_lambda_lst[k], rlr_y_test_est = sub_cross_validate_rlr(X, y, lambdas, partition, K=K)
        rlr_se = np.square(y_test - rlr_y_test_est).sum(axis=0)
        rlr_test_errors[k] = rlr_se / y_test.shape[0]

        opt_n_hu[k], ann_y_test_est = sub_cross_validate_ann(X, y, hidden_unit_options, partition, K=K, max_iter=5000)
        ann_se = np.square(ann_y_test_est - y_test).sum()  # squared error

        ann_test_errors[k] = ann_se / y_test.shape[0]  # Divide by number of observations

        # Baseline
        # using solely the mean of y_test to estimate , i.e. does not make use of the feature data
        baseline_test_errors[k] = np.square(y_test - y_test.mean(axis=0)).sum(axis=0) / y_test.shape[0]
    output = {"ANN": tuple(zip(opt_n_hu, ann_test_errors)),
              "RLR": tuple(zip(opt_lambda_lst, rlr_test_errors)),
              "BASE": baseline_test_errors}
    return output


def plot_models(X, y, opt_hidden_units, opt_lambda, K=10):
    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    train_index, test_index = list(ss.split(X))[0]
    partition = (train_index, test_index)
    rlr_err, yhatRLR = sub_cross_validate_rlr(X, y, lambdas=[opt_lambda], partition=partition, K=K)
    ann_err, yhatANN = sub_cross_validate_ann(X, y, hidden_unit_options=[opt_hidden_units], partition=partition, K=K,
                                              max_iter=10000)
    y_train = y[train_index]
    y_test = y[test_index]
    yhatBASE = y_train.mean(axis=0) * np.ones(np.shape(y_test))

    plot(y_test, yhatRLR, 'bo', alpha=0.35)  # RLR
    plot(y_test, yhatANN, 'ro', alpha=0.35)  # ANN
    plot(y_test, yhatBASE, '+-', color="orange")  # BASE
    plt.plot([0, 10], [0, 10], 'k--', )  # "PERFECT"
    xlabel("y_test (TRUE VALUES)")
    ylabel("y estimated (MODELS)")
    legend(["y_test vs {} estimate ".format(model_name) for model_name in ["rlr", "ann", "base", "perfect"]],
           loc="best")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    show()

    # 2 subplots:
    figure(1, figsize=(12, 8))
    subplot(1, 2, 1)
    plot(y_test, yhatRLR, 'bo')
    plt.plot([0, 10], [0, 10], 'k--', )  # "PERFECT"
    legend(["y_test vs RLR model", "Perfect line"])
    xlabel('y_test')
    ylabel('RLR esimate')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    grid()

    subplot(1, 2, 2)
    # title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
    plot(y_test, yhatANN, 'ro')
    plt.plot([0, 10], [0, 10], 'k--', )  # "PERFECT"
    legend(["y_test vs ANN model", "Perfect line"])
    xlabel('y_test')
    ylabel('ANN esimate')
    # legend(['Train error', 'Validation error'])
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    grid()
    show()


def statistic_comparison(X, y, opt_hidden_units, opt_lambda, K=10):
    ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

    train_index, test_index = list(ss.split(X))[0]
    partition = (train_index, test_index)
    y_test = y[test_index]
    y_train = y[train_index]

    _, yhatRLR = sub_cross_validate_rlr(X, y, lambdas=[opt_lambda], partition=partition, K=K)
    _, yhatANN = sub_cross_validate_ann(X, y, hidden_unit_options=[opt_hidden_units], partition=partition, K=K,
                                        max_iter=10000)
    yhatBASE = y_train.mean(axis=0) * np.ones(np.shape(y_test))

    # perform statistical comparison of the models
    # compute z with squared error.
    zRLR = np.abs(y_test - yhatRLR) ** 2

    # compute confidence interval of model A
    alpha = 0.05
    # CIA = st.t.interval(1 - alpha, df=len(zA) - 1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zANN = np.abs(y_test - yhatANN) ** 2

    zBASE = np.abs(y_test - yhatBASE) ** 2
    #
    z_dict = {
        "RLRvANN": zRLR - zANN,
        "RLRvBASE": zRLR - zBASE,
        "ANNvBASE": zANN - zBASE
    }

    # Z:
    Z_hat = {}
    for key, value in z_dict.items():
        Z_hat[key] = np.mean(value)

    # Confidence interval
    CIs = {}
    for key, value in z_dict.items():
        CIs[key] = st.t.interval(1 - alpha, len(value) - 1, loc=np.mean(value), scale=st.sem(value))
    """
        "RLRvANN" : ,
        "RLRvBASE" : st.t.interval(1 - alpha, len(zRLR_v_BASE) - 1, loc=np.mean(zRLR_v_BASE), scale=st.sem(zRLR_v_BASE)),
        "ANNvBASE" : st.t.interval(1 - alpha, len(zANN_v_BASE) - 1, loc=np.mean(zANN_v_BASE), scale=st.sem(zANN_v_BASE))
    """

    p_dict = {}
    for key, value in z_dict.items():
        p_dict[key] = 2 * st.t.cdf(-np.abs(np.mean(value)) / st.sem(value), df=len(value) - 1)  # p-value
    print(f"- - - - -\n"
          f"{z_dict = }\n"
          f"{Z_hat = }\n"
          f"{CIs = }\n"
          f"{p_dict = }"
          f"- - - - -\n")

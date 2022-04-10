"""
The "K-fold cross-validation for model selection" algorithm (Algorithm 5 in the book) is described in pseudo-code as:
    Require: K, the number of folds in the cross-validation loop
    Require: M1, . . . ,MS. The S different models to select between
    Ensure: Ms∗ the optimal model suggested by cross-validation
        for k = 1, . . . , K splits do
            Let D_train_k, D_test_k the k’th split of D
            for s = 1, . . . , S models do
                Train model Ms on the data D_train_k
                Let E_test_Ms,k be the test error of the model Ms when it is tested on D_test_k
            end for
        end for
        For each s compute: Eˆgen_Ms = SUM((N_test_k / N) * (E_test_Ms,k), k=1..K)
        Select the optimal model: s∗ = arg mins Eˆgen_Ms
        Ms∗ is now the optimal model suggested by cross-validation
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, subplot, plot, xlabel, ylabel, clim, title, show, semilogx, grid, loglog, legend
from sklearn import model_selection
import sklearn.linear_model as lm
from toolbox_02450 import feature_selector_lr, bmplot, rlr_validate, train_neural_net, draw_neural_net
import torch

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
            subplot(2, int(np.ceil(len(ff) / 2)), i + 1)
            plot(X[:, ff[i]], residual, '.')
            xlabel(attributeNames[ff[i]])
            ylabel('residual error')

    show()

def cross_validate_lambda(X, y, K, attributeNames, lambdas=np.power(10., range(-5, 9))):
    N, M = X.shape

    ## Crossvalidation
    # Create crossvalidation partition for evaluation
    CV = model_selection.KFold(K, shuffle=True)
    # CV = model_selection.KFold(K, shuffle=False)


    # Initialize variables
    # T = len(lambdas)
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

def cross_validate_ann(X, y, K, attributeNames):
    N, M = X.shape
    CV = model_selection.KFold(K, shuffle=True)
    
    # Parameters for neural network
    n_hidden_units = 2      # number of hidden units
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 1000
  

    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
    # Define the model
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index,:])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index,:])
        
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
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
        
        # Display the learning curve for the best net in the current fold
        h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
        h.set_label('CV fold {0}'.format(k+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')

    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, K+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')
        
    print('Diagram of best neural net in last fold:')
    weights = [net[i].weight.data.numpy().T for i in [0,2]]
    biases = [net[i].bias.data.numpy() for i in [0,2]]
    tf =  [str(net[i]) for i in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

    # Print the average classification error rate
    print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10,10))
    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Rating: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()

    plt.show()
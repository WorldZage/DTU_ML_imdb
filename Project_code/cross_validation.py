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


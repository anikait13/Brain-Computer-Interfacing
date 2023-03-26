# -*- coding: utf-8 -*-
"""Parameters for SVC, kNN and LDA, suitable for GridSearchCV from Sklearn."""

from sklearn.svm import SVC


def para_svc():
    """GridSearchCV parameteres for SVC."""

    para_svc = [{
        'C': [1000, 100, 10, 1],
        'cache_size': [1000],
        # 'class_weight': [],
        # 'coef0': [],
        'decision_function_shape': ['ovo'],
        # 'degree': [],
        'gamma': ['auto'],
        # 'kernel': ['linear','rbf']
        'kernel': ['rbf'],
        'max_iter': [1000, 5000, 10000, 100000, -1],
        # 'probability': [True],
        # 'random_state': [1000],
        'shrinking': [True, False],
        'tol': [1e-2, 1e-3, 1e-4]}
    ]
    return para_svc


def para_knn():
    """GridSearchCV parameters for kNN."""

    para_knn = [{
        'weights': ['distance'],
        'algorithm': ['brute'],
        'leaf_size': [5, 10, 15, 30],
        'p': [1, 2],
        # 'metric': ('minkowski'),
        # 'metric_params': (),
        # 'n_jobs': (None),
    },
        {
            'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20, 21, 25, 30, 31],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]}]
    return para_knn


def para_lda():
    """GridSearchCV parameteres for LDA."""
    1
    para_lda = [{
        'solver': ['svd'],
        # 'shrinkage': ['auto'],
        'n_components': [5, 10, 15, 20, 25, 30],
        # 'priors': [],
        # 'store_covariance': [],
        'tol': [1e-2, 1e-3, 1e-4]},
        {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': ['auto'],
            'n_components': [5, 10, 15, 20, 25, 30],
            'tol': [1e-2, 1e-3, 1e-4]}]
    return para_lda


def para_rfc():
    """GridSearchCV parameters for Random forest classifiers."""
    para_rfc = [{
        'n_estimators': [100, 200, 500, 50],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2'],
        # max_depth:
        # min_samples_split:
        # min_samples_leaf:
    }]

    return para_rfc


def para_mlpc():
    """GridSearchCV parameters for Multi Layer Perceptron."""
    para_mlpc = [{
        "hidden_layer_sizes": [60, 30, 10, 4],
        # 'activation': ['relu', 'tanh', 'sgd'],
        'activation': ['logistic', 'relu', 'identity'],
        'solver': ['lbfgs', 'adam'],
        'learning_rate': ['adaptive'],
        'warm_start': [False],
        'early_stopping': [True]
        # max_depth:
        # min_samples_split:
        # min_samples_leaf:
    }]

    return para_mlpc


def para_ada():
    """GridSearchCV parameters for ADA Boost algorithm."""
    para_ada = [{
        'estimator': [SVC(gamma='auto', kernel='rbf')],
        'learning_rate': [1.0],
        'algorithm': ['SAMME'],
        'n_estimators': [50],
        'random_state': [None]
    }]

    return para_ada

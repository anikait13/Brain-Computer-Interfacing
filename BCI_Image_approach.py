# -*- coding: utf-8 -*-
# version 1.1
"""Preprocessing and machine learning tools for the Kara One database
(openly available EEG data from an imagined speech research). 
"""

import glob
import mne
import scipy.io as sio
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy import signal as sig
from time import time
from features import fast_feat_array, pretty_feat_array
import csv


class Dataset:
    """Loading and preprocessing of the KARA ONE database.
    """
    registry = []
    figures_path = os.path.dirname(os.path.abspath(__file__)) + '/figures'
    os.makedirs(figures_path, exist_ok=True)

    def __init__(self, subject):
        self.name = subject
        self.registry.append(self)
        # TODO is self in self.registry neccesary 

    def load_data(self, path_to_data, raw=True):
        """Load subject data from the Kara One dataset.
            and then make image of each label
        """
        self.dataPath = path_to_data + self.name
        print(self.dataPath)
        os.chdir(self.dataPath)
        if raw:
            self.prompts = np.loadtxt("%s/Y.csv" % self.dataPath, dtype=str)
            print(self.prompts)
            print("Converting to 64 X 1172 matrix...")
            mat = sio.loadmat("%s/all_features_ICA.mat" % self.dataPath)

            for label in range(len(self.prompts)):
                label_mat = mat['all_features']['eeg_features'][0][0][0][0][0][0][label]
                with open('%s/Data/%s/%i.csv' % ('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing', self.name,
                                                 label), "w") as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    for line in label_mat:
                        writer.writerow(line)

        else:
            print("Loading filtered data.")
            for f in glob.glob("*-filtered.fif"):
                self.eeg_data = mne.io.read_raw_fif(f, 'standard_1020', preload=True)

    def find_best_features(self, feature_limit=30, single_channel=True):
        """Select n best features.

        Notes
        -----
        Reduces dimensionality and redundancy of features. 
        The implementation is based on Anova.

        Parameters
        ----------
        feature_limit : int
            Number of features to leave. 

        Returns
        -------
        X :ndarray
            2d array of signal features with 'float' type.
        Y : ndarray
            1D array of labels with 'str' type.
        """
        if single_channel:
            X = self.X
            Y = self.Y

            print("Calculating ANOVA.")
            selector = SelectKBest(score_func=f_classif, k=feature_limit)

            selector.fit(X['feature_value'], Y)
            chosen = []
            for idx in selector.get_support([True]):
                chosen.append(
                    [selector.scores_[idx], selector.pvalues_[idx], X[0, idx]['channel'], X[0, idx]['feature_name']])

            chosen.sort(key=lambda s: s[1])
            for chsn in chosen:
                print("F= %0.3f\tp = %0.3f\t channel = %s\t fname = %s" % (chsn[0], chsn[1], chsn[2], chsn[3]))

            trans_chosen = np.transpose(chosen)
            for chosen, text in (
                    (trans_chosen[2], 'Scored by channels: '),
                    (trans_chosen[3], 'Scored by features: ')):
                unique, counts = np.unique(chosen, return_counts=True)
                sorted_counts = sorted(dict(zip(unique, counts)).items(), reverse=True, key=lambda s: s[1])
                print(text, sorted_counts)

            print("ANOVA calculated, ", len(X[0]) - feature_limit, "features removed,", feature_limit,
                  " features left.")
            X = selector.transform(X)

            return X['feature_value'], Y, list(selector.get_support([True]))


class Classifier:
    """Prepare ML models and classify data.
    Notes
    -----
    The class provides methods for parameters optimisation
    and data classification. Former one utilise exhaustive search,
    the latter inputted classifiers in repeated stratified kfold model.

    Algorithm's parameters are not relevant if grid_search_sklearn() is
    to be used, adequate parameters' ranges should be inputted instead.

    Attributes
    ---------
    registry : list
        list of class instances.
    name : str
        name of classifier for logging purposes.
    algorithm : object
        classifier object.

    Methods
    -------
    grid_search_sklearn(self, X, Y, parameters)
        
    classify(self, X, Y, crval_splits=6, crval_repeats=10)

    """
    registry = []
    instances = 0

    def __init__(self, name, algorithm):
        self.registry.append(self)
        self.name = name
        self.algorithm = algorithm
        Classifier.instances += 1

    def grid_search_sklearn(self, X, Y, parameters):
        """Optimise classifier parameters using exhaustive search.

        Parameters
        ---------
        X, Y : array_like
            data for classifier in Sklearn-compatible format. 
        parameters : dict
            Dictionary of parameters for Sklearn.GridSearchCV.
        """
        print('-' * 80)
        print("Performing grid search for ", self.name, " algorithm...")
        grid_search = GridSearchCV(self.algorithm, parameters, n_jobs=-2, error_score=0, verbose=0)
        t0 = time()
        grid_search.fit(X, Y)
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best score: %0.3f" % grid_search.best_score_)
        best_parameters = grid_search.best_estimator_.get_params()
        self.algorithm.set_params(**best_parameters)
        print("Best parameters for ", self.name, ":\n", best_parameters)

    def classify(self, X, Y, crval_splits=6, crval_repeats=2):
        """Classify data.

        Notes
        -----
        Repeated stratified K-fold is a cross-validation model, 
        which repeats splitting of the data with a different
        randomization in each iteration. 

        Parameters
        ----------
        X, Y : array_like
            data for classifier in Sklearn-compatible format. 
        crval_splits : int
            Number of splits for cross-validation.
        crval_repeats : int
            Number of repeats for classification
        Returns
        -------
        Accuracy, F1 : list
            Accuracy and F scores from each pass, not averaged. 
        """
        Accuracy = []
        F1 = []
        CFM = []
        rsk = RepeatedStratifiedKFold(n_splits=crval_splits, n_repeats=crval_repeats)
        t0 = time()
        for train, test in rsk.split(X, Y):
            self.algorithm.fit(X[train], Y[train])
            predicted = self.algorithm.predict(X[test])
            Accuracy.append(accuracy_score(Y[test], predicted) * 100)
            F1.append(f1_score(Y[test], predicted, average='macro') * 100)
            print("we guessed", predicted)
            print("the answer was ", Y[test])
            print(confusion_matrix(Y[test], predicted))
            CFM.append(confusion_matrix(Y[test], predicted))
        print('-' * 40 + '\n%s\n' % self.name + '-' * 40)
        print("Parameters: ", self.algorithm.get_params())
        print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
        print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
        print("\nConfusion Matrix:\n", np.sum(CFM, axis=0), '\nNumber of instances: ', np.sum(CFM))
        print("done in %0.3fs" % (time() - t0))

        return Accuracy, F1, np, np.mean(Accuracy)


if __name__ == '__main__':
    pass

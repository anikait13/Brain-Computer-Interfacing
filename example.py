from gridSearchParameters import *
from expVariants import *
from BCI_Main import Dataset, Classifier
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

PATH_TO_DATA = "/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/Dataset/"

# Build instances of classifiers.
Classifier("Support Vector Machine", SVC())
# Classifier("Random Forest Classifier", RandomForestClassifier())
# Classifier("k-nearest neighbors", KNeighborsClassifier())
# Classifier("Linear Discriminant Analysis", LDA())
# Classifier("Neural Network", MLPClassifier())
# Classifier("ADA Boost", AdaBoostClassifier())


# list of experimental variants from expVariants.
mode_list = (mode1(), mode2(), mode3(), mode4(), mode5(), mode6())

# load parameters for grid search from gridSearchParameters.
# parameters_list = (para_svc(), para_knn(), para_lda(),para_mlpc(),para_rfc(),para_)
parameters_list = (para_svc())

#channels
eeg_channels = ('FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2','C4', 'C6', 'T8',
                'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
                'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ',
                'O2')
print(eeg_channels.index('FC4'))
# SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
#             'MM16', 'MM18', 'MM19', 'MM20', 'MM21')
SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10')

scores = []
# Iterate over subjects, preprocess the data and get scores.
overall_accuracy = []
features_arr = []

overall_accuracy = []
for test_subject in SUBJECTS:
    print(f"Testing subject {test_subject}")
    train_subjects = [subj for subj in SUBJECTS if subj != test_subject]
    print(train_subjects)
    scores = []
    features_arr = []
    for subject in train_subjects:
        Dataset(subject)
    first_run = True
    for subject in Dataset.registry:
        if first_run:
            first_run = False
            print("Training for : ", subject.name)
            raw = subject.load_data(PATH_TO_DATA, raw=True)
            subject.select_channels(channels=64)
            subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
            X, Y = subject.prepare_data(mode_list[1], scale_data=True)
        features_arr.append((X, Y))
        feature_chosen, X, Y = subject.find_best_features(feature_limit=20, single_channel=True)




    # Concatenate all features and labels from the training subjects
    X_train = np.concatenate([X for X, _ in features_arr], axis=0)
    Y_train = np.concatenate([Y for _, Y in features_arr], axis=0)
    subject.find_best_features(feature_limit=100, single_channel=True)

    print()
    # feature_chosen, X, Y = subject.find_best_features(X_train,feature_limit=1488, single_channel=True)
    # grid_search = GridSearchCV(SVC(), para_svc(), n_jobs=-2, error_score=0, verbose=0)
    # grid_search.fit(X_train, Y_train)
    # print()
    # print("Best score: %0.3f" % grid_search.best_score_)
    # best_parameters = grid_search.best_estimator_.get_params()
    # clf = SVC()
    # clf.set_params(**best_parameters)
    # print("Best parameters for SVC :\n", best_parameters)
    # clf.fit(X_train, Y_train)
    for idx, clf in enumerate(Classifier.registry):
        clf.grid_search_sklearn(X_train, Y_train, parameters_list[idx])
        print(parameters_list)

    for clf in Classifier.registry:
        Accuracy = []
        F1 = []
        CFM = []
        score = clf.classify(X_train, Y_train)
        overall_accuracy.append(score[3])
        print(score[3])
        Dataset(test_subject)
        subject = Dataset.registry[-1]
        raw = subject.load_data(PATH_TO_DATA, raw=True)
        subject.select_channels(channels=62)
        subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
        X_test, Y_test = subject.prepare_data(mode_list[1], scale_data=True)
        predicted = clf.predict(X_test)
        Accuracy.append(accuracy_score(Y_test, predicted) * 100)
        F1.append(f1_score(Y_test, predicted, average='macro') * 100)
        print("we guessed", predicted)
        print("the answer was ", Y_test)
        print(confusion_matrix(Y_test, predicted))
        print(Accuracy)
        CFM.append(confusion_matrix(Y_test, predicted))


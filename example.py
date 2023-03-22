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
# Classifier("Support Vector Machine", SVC())
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

# Full list of subjects from the study:


# MM09 outlier bad
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

    for subject in Dataset.registry:
        print("Training for : ", subject)
        raw = subject.load_data(PATH_TO_DATA, raw=False)
        subject.select_channels(channels=62)
        subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
        subject.prepare_data(mode_list[1], scale_data=True)
        X, Y = subject.find_best_features(feature_limit=30)
        features_arr.append((X, Y))

    # Concatenate all features and labels from the training subjects
    X_train = np.concatenate([X for X, _ in features_arr], axis=0)
    Y_train = np.concatenate([Y for _, Y in features_arr], axis=0)

    grid_search = GridSearchCV(SVC(), para_svc(), n_jobs=-2, error_score=0, verbose=0)
    grid_search.fit(X_train, Y_train)
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    clf = SVC()
    clf.set_params(**best_parameters)
    print("Best parameters for SVC :\n", best_parameters)
    clf.fit(X_train, Y_train)
    Accuracy = []
    F1 = []
    CFM = []
    # Test the classifier on the test subject
    Dataset(test_subject)
    subject = Dataset.registry[-1]
    raw = subject.load_data(PATH_TO_DATA, raw=True)
    subject.select_channels(channels=62)
    subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X_test, Y_test = subject.find_best_features(feature_limit=30)
    best_parameters = grid_search.best_estimator_.get_params()
    clf.fit(X_train, Y_train)
    predicted = clf.predict(X_test)
    Accuracy.append(accuracy_score(Y_test, predicted) * 100)
    F1.append(f1_score(Y_test, predicted, average='macro') * 100)
    print("we guessed", predicted)
    print("the answer was ", Y_test)
    print(confusion_matrix(Y_test, predicted))
    print(Accuracy)
    CFM.append(confusion_matrix(Y_test, predicted))

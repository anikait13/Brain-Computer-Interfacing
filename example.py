from gridSearchParameters import *
from expVariants import *
from BCI_Main_updated import Dataset, Classifier
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE

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
Classifier("Neural Network", MLPClassifier())
# Classifier("ADA Boost", AdaBoostClassifier())

# list of experimental variants from expVariants.
mode_list = (mode1(), mode2(), mode3(), mode4(), mode5(), mode6())

# load parameters for grid search from gridSearchParameters.
# parameters_list = (para_svc(), para_knn(), para_lda(),para_mlpc(),para_rfc(),para_)
parameters_list = (para_mlpc())

# channels eeg_channels = ('FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
# 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2','C4', 'C6', 'T8',
# 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2',
# 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2')

# SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
#             'MM16', 'MM18', 'MM19', 'MM20', 'MM21')

SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
            'MM16', 'MM18', 'MM19', 'MM20', 'MM21')

# Repeated Stratified K-Fold Validation

scores = []
overall_accuracy = []
features_arr = []
X_main = []
Y_main = []
# done to fasten algorithm for testing
features_arr = []
for i in range(1488):
    features_arr.append(i)

for subject in SUBJECTS:
    Dataset(subject)

for subject in Dataset.registry:
    print("Training for : ", subject.name)
    raw = subject.load_data(PATH_TO_DATA, raw=True)
    # subject.select_channels(channels=62)
    # subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
    # subject.prepare_data(mode_list[1], scale_data=True)
    # X, Y, selected_feats = subject.find_best_features(feature_limit=8, single_channel=True)
    # combined_list = list(set(features_arr + selected_feats))
    # features_arr = sorted(combined_list)
    # print(features_arr)

# done to fasten algorithm for testing

for subject in Dataset.registry:
    X, Y = subject.combine_features(features_arr)
    X_main.extend(X)
    Y_main.extend(Y)

X_main = np.array(X_main)
Y_main = np.hstack(Y_main)

print("X_train shape : ", X_main.shape)
print("Y_train shape : ", Y_main.shape)

estimator = SVC(kernel="linear")
estimator.get_params(deep=True)
selector = RFE(estimator, n_features_to_select=150, step=0.01)
selector.fit(X_main, Y_main)
arr = selector.support_
selected_features = []
for a in range(len(arr)):
    if arr[a]:
        selected_features.append(a)

X_main = selector.transform(X_main)

print("X_train shape : ", X_main.shape)

for idx, cl in enumerate(Classifier.registry):
    cl.grid_search_sklearn(X_main, Y_main, parameters_list[idx])
    print(parameters_list)

for cl in Classifier.registry:
    score = cl.classify(X_main, Y_main, crval_splits=13, crval_repeats=3)
    overall_accuracy.append(score[3])
    print(score[3])


for a in selected_features:
    print(a//24, a%24)



# Leave one subject out validation
#
# for test_subject in SUBJECTS:
#     print(f"Testing subject {test_subject}")
#     train_subjects = [subj for subj in SUBJECTS if subj != test_subject]
#     print(train_subjects)
#
#     scores = []
#     overall_accuracy = []
#     features_arr = []
#     X_train = []
#     Y_train = []
#
#     for subject in train_subjects:
#         Dataset(subject)
#
#     for subject in Dataset.registry:
#         print("Training for : ", subject.name)
#         raw = subject.load_data(PATH_TO_DATA, raw=True)
#         # subject.select_channels(channels=62)
#         # subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
#         # subject.prepare_data(mode_list[1], scale_data=True)
#         # X, Y, selected_feats = subject.find_best_features(feature_limit=10, single_channel=True)
#         # combined_list = list(set(features_arr + selected_feats))
#         # features_arr = sorted(combined_list)
#         # print(features_arr)

    # done to fasten algorithm for testing
    # features_arr = []
    # for i in range(1488):
    #     features_arr.append(i)
    #
    # for subject in Dataset.registry:
    #     X, Y = subject.combine_features(features_arr)
    #     X_train.extend(X)
    #     Y_train.extend(Y)
    #     print(Y_train)
    #
    # X_train = np.array(X_train)
    # Y_train = np.hstack(Y_train)
    #
    # print("X_train shape : ", X_train.shape)
    # print("Y_train shape : ", Y_train.shape)
    #
    # estimator = SVC(kernel="linear")
    # estimator.get_params(deep=True)
    # selector = RFE(estimator, n_features_to_select=100, step=0.01)
    # selector.fit(X_train, Y_train)
    # arr = selector.support_
    # selected_features = []
    # for a in range(len(arr)):
    #     if arr[a]:
    #         selected_features.append(a)
    # X_train = selector.transform(X_train)
    #
    # print("X_train shape : ", X_train.shape)
    # print("Y_train shape : ", Y_train.shape)
    #
    # grid_search = GridSearchCV(SVC(), para_svc(), n_jobs=-2, error_score=0, verbose=0)
    # grid_search.fit(X_train, Y_train)
    # print()
    # print("Best score: %0.3f" % grid_search.best_score_)
    # best_parameters = grid_search.best_estimator_.get_params()
    # clf = SVC()
    # clf.set_params(**best_parameters)
    # print("Best parameters for SVC :\n", best_parameters)
    # clf.fit(X_train, Y_train)
    #
    # # testing the fit
    # Accuracy = []
    # F1 = []
    # CFM = []
    # Dataset(test_subject)
    # subject = Dataset.registry[-1]
    # raw = subject.load_data(PATH_TO_DATA, raw=True)
    # # subject.select_channels(channels=62)
    # # subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
    # # X, Y = subject.prepare_data(mode_list[1], scale_data=True)
    # X_test, Y_test = subject.combine_features(selected_features)
    # predicted = clf.predict(X_test)
    # Accuracy.append(accuracy_score(Y_test, predicted) * 100)
    # F1.append(f1_score(Y_test, predicted, average='macro') * 100)
    # print("we guessed", predicted)
    # print("the answer was ", Y_test)
    # print(confusion_matrix(Y_test, predicted))
    # print("The Accuracy was", np.mean(Accuracy))

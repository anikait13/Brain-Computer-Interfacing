import numpy as np
from torch import save
import os
from expVariants import *
from BCI_Main_updated import Dataset, Classifier
from dbn.DBNAC import DBNClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

PATH_TO_DATA = "/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/Dataset/"

Classifier("Deep Belief Network", DBNClassifier())

# list of experimental variants from expVariants.
mode_list = (mode1(), mode2(), mode3(), mode4(), mode5(), mode6())

# Full list of subjects from the study: SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
# 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')

SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
            'MM16', 'MM18', 'MM19', 'MM20', 'MM21')

# initialize subjects' instances.
for subject in SUBJECTS:
    Dataset(subject)

scores = []
# Iterate over subjects, preprocess the data and get scores.


feature_scores = []

classifier = DBNClassifier(n_hiddens=[60], k=3,
                           loss_ae='MSELoss', loss_clf='CrossEntropyLoss',
                           optimizer_ae='Adam', optimizer_clf='Adam',
                           lr_rbm=1e-5, lr_ae=1e-2, lr_clf=1e-2,
                           epochs_rbm=100, epochs_ae=100, epochs_clf=100,
                           batch_size_rbm=100, batch_size_ae=200, batch_size_clf=200,
                           loss_ae_kwargs={}, loss_clf_kwargs={},
                           optimizer_ae_kwargs=dict(), optimizer_clf_kwargs=dict(),
                           random_state=42, use_gpu=True, verbose=False)

overall_accuracy = []
CFM = []
Accuracy = []
F1 = []
X_main = []
Y_main = []

for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=True)
    # subject.select_channels(channels=62)
    # subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
    # subject.prepare_data(mode_list[1], scale_data=True)
    # chosen, X, Y = subject.find_best_features(60)

# done to fasten algorithm for testing
features_arr = []
for i in range(1488):
    features_arr.append(i)

for subject in Dataset.registry:
    X, Y = subject.combine_features(features_arr)
    X_main.extend(X)
    Y_main.extend(Y)

X = np.array(X_main)
Y = np.hstack(Y_main)

# feature selection
# k-best
# selector = SelectKBest(score_func=f_classif, k=100)
# print(X)
# X = selector.fit_transform(X, Y)
# selected_features = []
# print(X.shape)
# print(Y)


# rfe using svm
selected_features = []
estimator = SVC(kernel="linear")
estimator.get_params(deep=True)
selector = RFE(estimator, n_features_to_select=100, step=0.01)
selector.fit(X, Y)
arr = selector.support_
for a in range(len(arr)):
    if arr[a]:
        selected_features.append(a)


X = selector.transform(X)



# amount of folds and repeats
rsk = RepeatedStratifiedKFold(n_splits=13, n_repeats=10)

for train, test in rsk.split(X, Y):
    classifier.fit(X[train], Y[train])
    predicted = classifier.predict(X[test])
    Accuracy.append(accuracy_score(Y[test], predicted) * 100)
    F1.append(f1_score(Y[test], predicted, average='macro') * 100)
    print("we guessed", predicted)
    print("the answer was ", Y[test])
    print(confusion_matrix(Y[test], predicted))
    print("Accuracy was : ", 100.0 * accuracy_score(Y[test], predicted))
    CFM.append(confusion_matrix(Y[test], predicted))

print('-' * 40 + '\n DBN Classifier Final \n' + '-' * 40)
print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
overall_accuracy.append(np.mean(Accuracy))
print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
print("\nConfusion Matrix:\n", np.sum(CFM, axis=0), '\nNumber of instances: ', np.sum(CFM))

print("Accuracy list ", overall_accuracy)
print("Accuracy: %.2f%%" % (np.mean(overall_accuracy)))
print(selected_features)


# One by one validation
# for subject in Dataset.registry:
#     subject.load_data(PATH_TO_DATA, raw=True)
#     subject.select_channels(channels=62)
#     subject.filter_data(lp_freq=50, hp_freq=1, save_filtered_data=True, plot=True)
#     subject.prepare_data(mode_list[1], scale_data=True)
#     chosen, X, Y = subject.find_best_features(60)
#     CFM = []
#     Accuracy = []
#     F1 = []
#
#     for train, test in rsk.split(X, Y):
#         classifier.fit(X[train], Y[train])
#         predicted = classifier.predict(X[test])
#         Accuracy.append(accuracy_score(Y[test], predicted) * 100)
#         F1.append(f1_score(Y[test], predicted, average='macro') * 100)
#         print("we guessed", predicted)
#         print("the answer was ", Y[test])
#         print(confusion_matrix(Y[test], predicted))
#         print("Accuracy was : ", 100.0 * accuracy_score(Y[test], predicted))
#         CFM.append(confusion_matrix(Y[test], predicted))
#
#     print('-' * 40 + '\n DBN Classifier Final \n' + '-' * 40)
#     print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
#     overall_accuracy.append(np.mean(Accuracy))
#     print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
#     print("\nConfusion Matrix:\n", np.sum(CFM, axis=0), '\nNumber of instances: ', np.sum(CFM))
#
# print("Accuracy list ", overall_accuracy)
# print("Accuracy: %.2f%%" % (np.mean(overall_accuracy)))
#
#

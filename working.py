from gridSearchParameters import *
from expVariants import *
from BCI_Main import Dataset, Classifier
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
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

# Full list of subjects from the study: SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
# 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')


#MM09 outlier bad
SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
            'MM16', 'MM18', 'MM19')

# initialize subjects' instances.
for subject in SUBJECTS:
    Dataset(subject)

scores = []
# Iterate over subjects, preprocess the data and get scores.
overall_accuracy = []
for subject in Dataset.registry:

    subject.load_data(PATH_TO_DATA, raw=True)
    subject.select_channels(channels=62)
    subject.filter_data(lp_freq=50, hp_freq=None, save_filtered_data=True, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X, Y = subject.find_best_features(feature_limit=25)

    for idx, cl in enumerate(Classifier.registry):
        cl.grid_search_sklearn(X, Y, parameters_list[idx])
        print(parameters_list)

    for cl in Classifier.registry:
        score = cl.classify(X, Y)
        overall_accuracy.append(score[3])
        print(score[3])


print("Accuracy list ", overall_accuracy)
print("Accuracy: %.2f%%" % (np.mean(overall_accuracy)))
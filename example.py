from gridSearchParameters import *
from expVariants import *
from BCI_Main import Dataset, Classifier

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
parameters_list = (para_svc() , para_ada())

# Full list of subjects from the study: SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
# 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')
SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12')

# initialize subjects' instances.
for subject in SUBJECTS:
    Dataset(subject)

scores = []
# Iterate over subjects, preprocess the data and get scores.
for subject in Dataset.registry:

    subject.load_data(PATH_TO_DATA, raw=False)
    subject.select_channels(channels=60)
    subject.filter_data(lp_freq=None, hp_freq=1, save_filtered_data=False, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X, Y = subject.find_best_features(feature_limit=20)

    for idx, cl in enumerate(Classifier.registry):
        cl.grid_search_sklearn(X, Y, parameters_list[idx])

    for cl in Classifier.registry:
        score = cl.classify(X, Y)
        scores.append(score)

print(scores[3])


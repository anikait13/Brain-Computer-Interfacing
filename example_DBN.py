from gridSearchParameters import *
from expVariants import *
from BCI_Main import Dataset, Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM

from dbn.tensorflow import SupervisedDBNClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import sys

PATH_TO_DATA = "/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/Dataset/"

# Build instances of classifiers.
# Classifier("Random Forest Classifier", RandomForestClassifier())
# Classifier("k-nearest neighbors", KNeighborsClassifier())
# Classifier("Linear Discriminant Analysis", LDA())
# Classifier("Neural Network", MLPClassifier())
Classifier("Deep Belief Network", SupervisedDBNClassification())



# list of experimental variants from expVariants.
mode_list = (mode1(), mode2(), mode3(), mode4(), mode5(), mode6())

# load parameters for grid search from gridSearchParameters.
# parameters_list = (para_svc(), para_knn(), para_lda())
parameters_list = (para_dbn())

# Full list of subjects from the study: SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
# 'MM16', 'MM18', 'MM19', 'MM20', 'MM21', 'P02')
SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12')

# initialize subjects' instances.
for subject in SUBJECTS:
    Dataset(subject)

scores = []
# Iterate over subjects, preprocess the data and get scores.
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                                 learning_rate_rbm=0.05,
                                                 learning_rate=0.1,
                                                 n_epochs_rbm=10,
                                                 n_iter_backprop=100,
                                                 batch_size=32,
                                                 activation_function='relu',
                                                 dropout_p=0.2)
for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=False)
    subject.select_channels(channels=60)
    subject.filter_data(lp_freq=None, hp_freq=1, save_filtered_data=False, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X, Y = subject.find_best_features(feature_limit=500)
    CFM = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    classifier.fit(X_train, Y_train)
    classifier.save('DBN_model.pkl')
    classifier = SupervisedDBNClassification.load('DBN_model.pkl')
    Y_pred = classifier.predict(X_test)
    print("Model predicted :", Y_pred)
    print("Real Value was :", Y_test)
    print(confusion_matrix(Y_test, Y_pred))
    CFM.append(confusion_matrix(Y_test, Y_pred))
    print('Done.\nAccuracy: ', 100.0 * accuracy_score(Y_test, Y_pred), "%")
    print(CFM)


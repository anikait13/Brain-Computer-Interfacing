import numpy as np
from torch import save

from expVariants import *
from BCI_Main import Dataset, Classifier
from dbn.DBNAC import DBNClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

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


# amount of folds and repeats
rsk = RepeatedStratifiedKFold(n_splits=6, n_repeats=5)

feature_scores = []

classifier = DBNClassifier(n_hiddens=[62], k=3,
                           loss_ae='MSELoss', loss_clf='CrossEntropyLoss',
                           optimizer_ae='Adam', optimizer_clf='Adam',
                           lr_rbm=1e-5, lr_ae=1e-2, lr_clf=1e-2,
                           epochs_rbm=200, epochs_ae=100, epochs_clf=100,
                           batch_size_rbm=100, batch_size_ae=100, batch_size_clf=100,
                           loss_ae_kwargs={}, loss_clf_kwargs={},
                           optimizer_ae_kwargs=dict(), optimizer_clf_kwargs=dict(),
                           random_state=42, use_gpu=True, verbose=False)

overall_accuracy = []

for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=True)
    subject.select_channels(channels=62)
    subject.filter_data(lp_freq=1, hp_freq=50, save_filtered_data=True, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X, Y = subject.find_best_features(93)
    CFM = []
    Accuracy = []
    F1 = []

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



import numpy as np

from expVariants import *
from BCI_Main import Dataset, Classifier
from dbn.DBNAC import DBNClassifier

# from dbn.tensorflow import SupervisedDBNClassification

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
SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12')

# initialize subjects' instances.
for subject in SUBJECTS:
    Dataset(subject)

scores = []
# Iterate over subjects, preprocess the data and get scores.

rsk = RepeatedStratifiedKFold(n_splits=6, n_repeats=5)

for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=True)
    subject.select_channels(channels=60)
    subject.filter_data(lp_freq=None, hp_freq=1, save_filtered_data=False, plot=True)
    subject.prepare_data(mode_list[1], scale_data=True)
    X, Y = subject.find_best_features(feature_limit=100)
    CFM = []
    Accuracy = []
    F1 = []

    classifier = DBNClassifier(n_hiddens=[25], k=3,
                               loss_ae='MSELoss', loss_clf='CrossEntropyLoss',
                               optimizer_ae='Adam', optimizer_clf='Adam',
                               lr_rbm=1e-5, lr_ae=0.01, lr_clf=0.01,
                               epochs_rbm=100, epochs_ae=50, epochs_clf=50,
                               batch_size_rbm=50, batch_size_ae=50, batch_size_clf=50,
                               loss_ae_kwargs={}, loss_clf_kwargs={},
                               optimizer_ae_kwargs=dict(), optimizer_clf_kwargs=dict(),
                               random_state=42, use_gpu=True, verbose=True, )

    for train, test in rsk.split(X, Y):
        # if first_iter:
        #     print("First Iteration:")
        #     classifier = SupervisedDBNClassification(hidden_layers_structure=[5, 5],
        #                                              learning_rate_rbm=0.05,
        #                                              learning_rate=1e-1,
        #                                              n_epochs_rbm=10,
        #                                              n_iter_backprop=100,
        #                                              batch_size=32,
        #                                              activation_function='relu',
        #                                              dropout_p=0.2)
        #     classifier.fit(X[train], Y[train])
        #     classifier.save('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/DBN_model.pkl')
        #     first_iter = False
        # else:
        #     print("Not First Iteration:")
        #     classifier = SupervisedDBNClassification.load('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing'
        #                                                   '/DBN_model.pkl')
        #     classifier.fit(X[train], Y[train], '/Users/anikait/Desktop/builds/Brain-Computer-Interfacing'
        #                                        '/DBN_model.pkl')
        # classifier.save('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/DBN_model.pkl')
        # # testing
        # classifier_testing = SupervisedDBNClassification.load('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing'
        #                                                       '/DBN_model.pkl')
        # predicted = classifier_testing.predict(X[test])
        #
        # # testing
        # classifier_testing = SupervisedDBNClassification.load('/Users/anikait/Desktop/builds/Brain-Computer-Interfacing'
        #                                                       '/DBN_model.pkl')
        # predicted = classifier_testing.predict(X[test])
        # Accuracy.append(accuracy_score(Y[test], predicted) * 100)
        # F1.append(f1_score(Y[test], predicted, average='macro') * 100)
        # print("we guessed", predicted)
        # print("the answer was ", Y[test])
        # print(confusion_matrix(Y[test], predicted))
        # CFM.append(confusion_matrix(Y[test], predicted))
        # print('Done.\nAccuracy: %f' % accuracy_score(Y[test], predicted))
        # print('-' * 40 + '\n%DBN Classifier\n' + '-' * 40)
        # print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
        # print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
        # print("\nConfusion Matrix:\n", np.sum(CFM, axis=0), '\nNumber of instances: ', np.sum(CFM))
        classifier.fit(X[train], Y[train])
        predicted = classifier.predict(X[test])
        Accuracy.append(accuracy_score(Y[test], predicted) * 100)
        F1.append(f1_score(Y[test], predicted, average='macro') * 100)
        print("we guessed", predicted)
        print("the answer was ", Y[test])
        print(confusion_matrix(Y[test], predicted))
        CFM.append(confusion_matrix(Y[test], predicted))

    print('-' * 40 + '\n%DBN Classifier Final \n' + '-' * 40)
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(Accuracy), np.std(Accuracy)))
    print("F1 score: %.2f%% (+/- %.2f%%)" % (np.mean(F1), np.std(F1)))
    print("\nConfusion Matrix:\n", np.sum(CFM, axis=0), '\nNumber of instances: ', np.sum(CFM))

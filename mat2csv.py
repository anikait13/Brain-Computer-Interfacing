import scipy.io as sio
import pandas as pd
import numpy as np
import karaone
from BCI_Main import Dataset, Classifier




test_subject = "MM05"
Dataset(test_subject)
print(Dataset.registry)
mat = sio.loadmat('Dataset/MM05/all_features_simple.mat')

features = mat['all_features'][0]['feature_labels'][0]
print(features)


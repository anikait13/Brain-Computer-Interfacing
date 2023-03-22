import scipy.io as sio
import pandas as pd
import numpy as np
import karaone

mat = sio.loadmat('Dataset/MM05/all_features_simple.mat')

features = mat['all_features']['prompts']
print(features.dtype)


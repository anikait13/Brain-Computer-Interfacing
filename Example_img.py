import scipy.io as sio
import pandas as pd
import numpy as np

from BCI_Image_approach import Dataset, Classifier

SUBJECTS = ('MM05', 'MM08', 'MM09', 'MM10', 'MM11', 'MM12', 'MM14', 'MM15',
            'MM16', 'MM18', 'MM19', 'MM20', 'MM21')

PATH_TO_DATA = "/Users/anikait/Desktop/builds/Brain-Computer-Interfacing/Dataset/"

for subject in SUBJECTS:
    Dataset(subject)

counter_knew = 0
counter_gnaw = 0
counter_pat = 0
counter_pot = 0
for subject in Dataset.registry:
    subject.load_data(PATH_TO_DATA, raw=False)  # load raw data and convert to csv
    counter_knew, counter_gnaw, counter_pat, counter_pot =\
        subject.csvtoimage(counter_knew, counter_gnaw, counter_pat, counter_pot)  # Convert csv to image

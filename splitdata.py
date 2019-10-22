# splitdata.py

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

class SplitData:

    sss = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state=1)
    strat_base_colname = 'median_income'
    bins = [0., 1.5, 3.0, 4.5, 6, np.inf]

    @staticmethod
    def getStratifiedSplit(data, strat_base_colname = strat_base_colname, bins = bins):
        labels = list(range(1,len(bins)))
        strat_colname = strat_base_colname + '_cat'
        data[strat_colname] = pd.cut(data[strat_base_colname], bins=bins, labels=labels)
        for train_idx, test_idx in SplitData.sss.split(data, data[strat_colname]):
            train_set = data.loc[train_idx]
            test_set = data.loc[test_idx]
        plit_params = {'strat_base_colname': strat_base_colname, 'bins': bins, 'labels': labels, 'strat_colname':strat_colname}
        return train_set, test_set, plit_params

    @staticmethod
    def get_random_split(data):
        return train_test_split(data)


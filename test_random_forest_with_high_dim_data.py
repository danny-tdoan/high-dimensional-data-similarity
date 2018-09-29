"""Experiment the similarity in high dimensional data based on RandomForest. The idea is:
1. Mark the original data to have label 1
2. Inject synthetic data with normal distribution (and should be separable from original data)
3. Mark the synthetic data with label 2
4. Fit a RandomForest on the entire data. The classifier should easily be able to separate label 1 and label 2.
5. Analyze the trees, the samples that usually end up in the same branch of a tree are more likely to be similar
6. The important of features can also be used to reveal important features (for similarity)

Check this paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3516432/
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

orig_data = pd.read_csv('data/input_500x20.csv')
orig_size = orig_data.shape

'''add new synthetic data just shuffle each column of the input to make sure the distribution is the same
but the dependence is lost'''
synthetic_data = shuffle(orig_data)


data = np.vstack((orig_data, synthetic_data))
label = np.ravel(np.concatenate((np.ones((orig_size[0], 1)), 2 * np.ones((synthetic_data.shape[0], 1))), axis=0))

clf = RandomForestClassifier(n_estimators=20)
clf.fit(data, label)

print(clf.feature_importances_)

import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA

pca = PCA()

X = np.load('../data/x_foreground_fold1_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold1_fg.npy", basis)

X = np.load('../data/x_foreground_fold2_train.npy')

pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold2_fg.npy", basis)

X = np.load('../data/x_foreground_fold3_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold3_fg.npy", basis)

X = np.load('../data/x_foreground_fold4_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold4_fg.npy", basis)


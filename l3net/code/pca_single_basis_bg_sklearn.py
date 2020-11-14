import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA

pca = PCA()

X = np.load('../data/x_background_fold1_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold1_bg.npy", basis)

X = np.load('../data/x_background_fold2_train.npy')

pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold2_bg.npy", basis)

X = np.load('../data/x_background_fold3_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold3_bg.npy", basis)

X = np.load('../data/x_background_fold4_train.npy')
pca.fit(X)

basis = pca.components_
np.save("../basis/basis_single_fold4_bg.npy", basis)


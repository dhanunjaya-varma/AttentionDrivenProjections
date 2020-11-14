import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics


def pca(data,nDim=0):
    # Centre data
    m = np.mean(data,axis=0)
    data = data - m
    
    # Covariance matrix
    C = np.cov(np.transpose(data))
    
    # Compute eigenvalues and sort into descending order
    evals,evecs = np.linalg.eig(C)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    if nDim>0:
        evecs = evecs[:,:nDim]
    # Produce the new data matrix
    evals = np.real(evals)
    evecs = np.real(evecs)
    #x = np.dot(np.transpose(evecs),np.transpose(data))
    # Compute the original data again
    #y=np.transpose(np.dot(evecs,x))+m
    return evecs

X = np.load('../data/x_foreground_fold1_train.npy')
Y = np.load('../data/y_foreground_fold1_train.npy')


basis = []

for i in range(15):
    print(i)
    idx_label = np.where(Y == i)
    data = []
    for j in range(len(idx_label[0])):
        data.append(X[idx_label[0][j]])
    data1 = np.array(data)
    print(data1.shape)
    basis.append(pca(data1))

basis = np.asarray(basis)
print(basis.shape)
np.save("../basis/basis_full_fold1_fg.npy", basis)

X = np.load('../data/x_foreground_fold2_train.npy')
Y = np.load('../data/y_foreground_fold2_train.npy')


basis = []

for i in range(15):
    print(i)
    idx_label = np.where(Y == i)
    data = []

    for j in range(len(idx_label[0])):
        data.append(X[idx_label[0][j]])
    data1 = np.array(data)
    print(data1.shape)
    basis.append(pca(data1))

basis = np.asarray(basis)
print(basis.shape)
np.save("../basis/basis_full_fold2_fg.npy", basis)

X = np.load('../data/x_foreground_fold3_train.npy')
Y = np.load('../data/y_foreground_fold3_train.npy')


basis = []

for i in range(15):
    print(i)
    idx_label = np.where(Y == i)
    data = []

    for j in range(len(idx_label[0])):
        data.append(X[idx_label[0][j]])
    data1 = np.array(data)
    print(data1.shape)
    basis.append(pca(data1))

basis = np.asarray(basis)
print(basis.shape)
np.save("../basis/basis_full_fold3_fg.npy", basis)

X = np.load('../data/x_foreground_fold4_train.npy')
Y = np.load('../data/y_foreground_fold4_train.npy')


basis = []

for i in range(15):
    print(i)
    idx_label = np.where(Y == i)
    data = []

    for j in range(len(idx_label[0])):
        data.append(X[idx_label[0][j]])
    data1 = np.array(data)
    print(data1.shape)
    basis.append(pca(data1))

basis = np.asarray(basis)
print(basis.shape)
np.save("../basis/basis_full_fold4_fg.npy", basis)
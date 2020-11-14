import os
from os.path import abspath
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import svm
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
#from sklearn.externals import joblib
from sklearn import metrics



cvscores = []  
X_train = np.load('../data/x_audio_fold1_train.npy')
Y_train = np.load('../data/y_audio_fold1_train.npy')

X_test = np.load('../data/x_audio_fold1_test.npy')
Y_test = np.load('../data/y_audio_fold1_test.npy')

    
svm_clas1 = svm.SVC(kernel='linear', C=0.01)

svm_clas1.fit(X_train, Y_train) 
#joblib.dump(svm_clas1,'svm_background_fold1_weights.pkl')

y_pred = svm_clas1.predict(X_test)
cvscores.append(metrics.accuracy_score(Y_test, y_pred)*100)
print("Accuracy fold 1:",metrics.accuracy_score(Y_test, y_pred)*100,"%")

X_train = np.load('../data/x_audio_fold2_train.npy')
Y_train = np.load('../data/y_audio_fold2_train.npy')

X_test = np.load('../data/x_audio_fold2_test.npy')
Y_test = np.load('../data/y_audio_fold2_test.npy')

    
svm_clas2 = svm.SVC(kernel='linear', C=0.01)

svm_clas2.fit(X_train, Y_train) 
#joblib.dump(svm_clas1,'svm_background_fold1_weights.pkl')

y_pred = svm_clas2.predict(X_test)
cvscores.append(metrics.accuracy_score(Y_test, y_pred)*100)
print("Accuracy fold 2:",metrics.accuracy_score(Y_test, y_pred)*100,"%")

X_train = np.load('../data/x_audio_fold3_train.npy')
Y_train = np.load('../data/y_audio_fold3_train.npy')

X_test = np.load('../data/x_audio_fold3_test.npy')
Y_test = np.load('../data/y_audio_fold3_test.npy')

    
svm_clas3 = svm.SVC(kernel='linear', C=0.01)

svm_clas3.fit(X_train, Y_train) 
#joblib.dump(svm_clas1,'svm_background_fold1_weights.pkl')

y_pred = svm_clas3.predict(X_test)
cvscores.append(metrics.accuracy_score(Y_test, y_pred)*100)
print("Accuracy fold 3:",metrics.accuracy_score(Y_test, y_pred)*100,"%")

X_train = np.load('../data/x_audio_fold4_train.npy')
Y_train = np.load('../data/y_audio_fold4_train.npy')

X_test = np.load('../data/x_audio_fold4_test.npy')
Y_test = np.load('../data/y_audio_fold4_test.npy')

    
svm_clas4 = svm.SVC(kernel='linear', C=0.01)

svm_clas4.fit(X_train, Y_train) 
#joblib.dump(svm_clas1,'svm_background_fold1_weights.pkl')

y_pred = svm_clas4.predict(X_test)
cvscores.append(metrics.accuracy_score(Y_test, y_pred)*100)
print("Accuracy fold 4:",metrics.accuracy_score(Y_test, y_pred)*100,"%")

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
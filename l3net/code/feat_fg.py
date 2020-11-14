from keras import models
from keras import layers
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cvscores = []

x_train = np.load('../data/x_train_auto_fold1_fg.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_test_auto_fold1_fg.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')

m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m


model1 = models.load_model('../models/fold1_att_model_fg.h5')
model = model1.layers[6].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold1:", metrics.accuracy_score(y_test, y_pred)*100)


x_train = np.load('../data/x_train_auto_fold2_fg.npy')
y_train = np.load('../data/y_audio_fold2_train.npy')
x_test = np.load('../data/x_test_auto_fold2_fg.npy')
y_test = np.load('../data/y_audio_fold2_test.npy')

m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m


model1 = models.load_model('../models/fold2_att_model_fg.h5')
model = model1.layers[6].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold2:", metrics.accuracy_score(y_test, y_pred)*100)

x_train = np.load('../data/x_train_auto_fold3_fg.npy')
y_train = np.load('../data/y_audio_fold3_train.npy')
x_test = np.load('../data/x_test_auto_fold3_fg.npy')
y_test = np.load('../data/y_audio_fold3_test.npy')

m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

model1 = models.load_model('../models/fold3_att_model_fg.h5')
model = model1.layers[6].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold3:", metrics.accuracy_score(y_test, y_pred)*100)

x_train = np.load('../data/x_train_auto_fold4_fg.npy')
y_train = np.load('../data/y_audio_fold4_train.npy')
x_test = np.load('../data/x_test_auto_fold4_fg.npy')
y_test = np.load('../data/y_audio_fold4_test.npy')

m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

model1 = models.load_model('../models/fold4_att_model_fg.h5')
model = model1.layers[6].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
print("Accuracy Fold4:", metrics.accuracy_score(y_test, y_pred)*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

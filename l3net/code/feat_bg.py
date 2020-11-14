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
evalscores = []

x_train = np.load('../data/x_train_auto_fold1_bg.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_test_auto_fold1_bg.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')
#x_eval = np.load('./x_eval_auto_fold1_bg.npy') 
#y_eval = np.load('./y_audio_eval_train.npy')


model1 = models.load_model('../models/fold1_att_model_bg.h5')
model = model1.layers[4].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

#pred_eval = model.predict(x_eval)
#print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
#y_pred_eval = svm_clas.predict(pred_eval)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
#evalscores.append(metrics.accuracy_score(y_eval, y_pred_eval)*100)
print("Accuracy Fold1:", metrics.accuracy_score(y_test, y_pred)*100)
#print("Accuracy Fold1:", metrics.accuracy_score(y_eval, y_pred_eval)*100)


x_train = np.load('../data/x_train_auto_fold2_bg.npy')
y_train = np.load('../data/y_audio_fold2_train.npy')
x_test = np.load('../data/x_test_auto_fold2_bg.npy')
y_test = np.load('../data/y_audio_fold2_test.npy')

#x_eval = np.load('./x_eval_auto_fold2_bg.npy') 
#y_eval = np.load('./y_audio_eval_train.npy')


model1 = models.load_model('../models/fold2_att_model_bg.h5')
model = model1.layers[4].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

#pred_eval = model.predict(x_eval)
#print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
#np.save('fold2_att_pred.npy', y_pred)
#y_pred_eval = svm_clas.predict(pred_eval)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
#evalscores.append(metrics.accuracy_score(y_eval, y_pred_eval)*100)
print("Accuracy Fold2:", metrics.accuracy_score(y_test, y_pred)*100)
#print("Accuracy Fold2:", metrics.accuracy_score(y_eval, y_pred_eval)*100)

x_train = np.load('../data/x_train_auto_fold3_bg.npy')
y_train = np.load('../data/y_audio_fold3_train.npy')
x_test = np.load('../data/x_test_auto_fold3_bg.npy')
y_test = np.load('../data/y_audio_fold3_test.npy')
#x_eval = np.load('./x_eval_auto_fold3_bg.npy') 
#y_eval = np.load('./y_audio_eval_train.npy')


model1 = models.load_model('../models/fold3_att_model_bg.h5')
model = model1.layers[4].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

#pred_eval = model.predict(x_eval)
#print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
#y_pred_eval = svm_clas.predict(pred_eval)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
#evalscores.append(metrics.accuracy_score(y_eval, y_pred_eval)*100)
print("Accuracy Fold3:", metrics.accuracy_score(y_test, y_pred)*100)
#print("Accuracy Fold3:", metrics.accuracy_score(y_eval, y_pred_eval)*100)

x_train = np.load('../data/x_train_auto_fold4_bg.npy')
y_train = np.load('../data/y_audio_fold4_train.npy')
x_test = np.load('../data/x_test_auto_fold4_bg.npy')
y_test = np.load('../data/y_audio_fold4_test.npy')
#x_eval = np.load('./x_eval_auto_fold4_bg.npy') 
#y_eval = np.load('./y_audio_eval_train.npy')


model1 = models.load_model('../models/fold4_att_model_bg.h5')
model = model1.layers[4].output
model = models.Model(inputs=model1.input, outputs=model)

pred_train = model.predict(x_train)
print(pred_train.shape)

pred_test = model.predict(x_test)
print(pred_test.shape)

#pred_eval = model.predict(x_eval)
#print(pred_test.shape)

svm_clas = svm.SVC(kernel='linear', C=0.01)

svm_clas.fit(pred_train, y_train) 
#joblib.dump(svm_clas1,'svm_error_fold1_weights.pkl')

y_pred = svm_clas.predict(pred_test)
#y_pred_eval = svm_clas.predict(pred_eval)
cvscores.append(metrics.accuracy_score(y_test, y_pred)*100)
#evalscores.append(metrics.accuracy_score(y_eval, y_pred_eval)*100)
print("Accuracy Fold4:", metrics.accuracy_score(y_test, y_pred)*100)
#print("Accuracy Fold4:", metrics.accuracy_score(y_eval, y_pred_eval)*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
#print("%.2f%% (+/- %.2f%%)" % (np.mean(evalscores), np.std(evalscores)))

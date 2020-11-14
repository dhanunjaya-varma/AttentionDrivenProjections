import numpy as np
import tensorflow as tf
import random as rn

np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import os
import sys
#import numpy as np
from sklearn import metrics

import keras
from keras.models import Model
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical

try:
    import cPickle
except BaseException:
    import _pickle as cPickle
from sklearn.model_selection import train_test_split

def attention_pooling(inputs, **kwargs):
    [out, att] = inputs

    epsilon = 1e-7
    att = K.clip(att, epsilon, 1. - epsilon)
    normalized_att = att / K.sum(att, axis=1)[:, None, :]

    return K.sum(out * normalized_att, axis=1)


def pooling_shape(input_shape):
	(sample_num, time_steps, freq_bins) = input_shape
	return (sample_num, freq_bins)

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)



hidden_units = 256
classes_num = 15
drop_rate = 0.5
x_train = np.load('../data/x_train_auto_fold1_fg.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_test_auto_fold1_fg.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')

m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=3)
input_layer = Input(shape=(x_train.shape[1],x_train.shape[2]))
#a3 = Dense(encoding_dim, activation='relu')(input_layer)
cla = Dense(hidden_units, activation='linear')(input_layer)
att = Dense(hidden_units, activation='sigmoid')(input_layer)
#att = Activation(activation='sigmoid')(cla)
b1 = Lambda(attention_pooling, output_shape=(hidden_units,))([cla, att])

b1 = BatchNormalization()(b1)
b1 = Activation(activation='relu')(b1)
b1 = Dropout(drop_rate)(b1)

output_layer = Dense(classes_num, activation='softmax')(b1)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=16,
                        epochs=15, verbose=1, shuffle=False)
scores = model.evaluate(x_test, y_test, verbose=0, batch_size=16)
print("Accuracy fold1: ", scores[1]*100)
model.save('../models/fold1_att_model_fg.h5')

np.load = np_load_old
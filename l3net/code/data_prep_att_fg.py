import numpy as np


p = 0
q = 200
x_train = np.load('../data/x_audio_fold1_train.npy')
y_train = np.load('../data/y_audio_fold1_train.npy')
x_test = np.load('../data/x_audio_fold1_test.npy')
y_test = np.load('../data/y_audio_fold1_test.npy')
basis = np.load('../basis/basis_full_fold1_fg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

x_train_new = []
x_test_new = []
for i in range(15):
	mat_fg = np.dot(basis[i][:, p:q], np.transpose(basis[i][:, p:q]))
	x_train_new.append(np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train))))
	x_test_new.append(np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test))))

x_train_new = np.array(x_train_new)
x_test_new = np.array(x_test_new)

x_train_new = np.swapaxes(x_train_new, 0, 1)
x_test_new = np.swapaxes(x_test_new, 0, 1)

print(x_train_new.shape)
print(x_test_new.shape)

np.save('../data/x_train_auto_fold1_fg.npy', x_train_new)
np.save('../data/x_test_auto_fold1_fg.npy', x_test_new)


x_train = np.load('../data/x_audio_fold2_train.npy')
y_train = np.load('../data/y_audio_fold2_train.npy')
x_test = np.load('../data/x_audio_fold2_test.npy')
y_test = np.load('../data/y_audio_fold2_test.npy')
basis = np.load('../basis/basis_full_fold2_fg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

x_train_new = []
x_test_new = []
for i in range(15):
	mat_fg = np.dot(basis[i][:, p:q], np.transpose(basis[i][:, p:q]))
	x_train_new.append(np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train))))
	x_test_new.append(np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test))))

x_train_new = np.array(x_train_new)
x_test_new = np.array(x_test_new)

x_train_new = np.swapaxes(x_train_new, 0, 1)
x_test_new = np.swapaxes(x_test_new, 0, 1)

print(x_train_new.shape)
print(x_test_new.shape)

np.save('../data/x_train_auto_fold2_fg.npy', x_train_new)
np.save('../data/x_test_auto_fold2_fg.npy', x_test_new)

x_train = np.load('../data/x_audio_fold3_train.npy')
y_train = np.load('../data/y_audio_fold3_train.npy')
x_test = np.load('../data/x_audio_fold3_test.npy')
y_test = np.load('../data/y_audio_fold3_test.npy')
basis = np.load('../basis/basis_full_fold3_fg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

x_train_new = []
x_test_new = []
for i in range(15):
	mat_fg = np.dot(basis[i][:, p:q], np.transpose(basis[i][:, p:q]))
	x_train_new.append(np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train))))
	x_test_new.append(np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test))))

x_train_new = np.array(x_train_new)
x_test_new = np.array(x_test_new)

x_train_new = np.swapaxes(x_train_new, 0, 1)
x_test_new = np.swapaxes(x_test_new, 0, 1)

print(x_train_new.shape)
print(x_test_new.shape)

np.save('../data/x_train_auto_fold3_fg.npy', x_train_new)
np.save('../data/x_test_auto_fold3_fg.npy', x_test_new)

x_train = np.load('../data/x_audio_fold4_train.npy')
y_train = np.load('../data/y_audio_fold4_train.npy')
x_test = np.load('../data/x_audio_fold4_test.npy')
y_test = np.load('../data/y_audio_fold4_test.npy')
basis = np.load('../basis/basis_full_fold4_fg.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

x_train_new = []
x_test_new = []
for i in range(15):
	mat_fg = np.dot(basis[i][:, p:q], np.transpose(basis[i][:, p:q]))
	x_train_new.append(np.transpose(np.transpose(x_train) - np.dot(mat_fg,np.transpose(x_train))))
	x_test_new.append(np.transpose(np.transpose(x_test) - np.dot(mat_fg,np.transpose(x_test))))

x_train_new = np.array(x_train_new)
x_test_new = np.array(x_test_new)

x_train_new = np.swapaxes(x_train_new, 0, 1)
x_test_new = np.swapaxes(x_test_new, 0, 1)

print(x_train_new.shape)
print(x_test_new.shape)

np.save('../data/x_train_auto_fold4_fg.npy', x_train_new)
np.save('../data/x_test_auto_fold4_fg.npy', x_test_new)
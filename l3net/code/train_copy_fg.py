import shutil
import numpy as np


classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

with open('evaluation_setup/fold1_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/foreground/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_foreground_fold1_train.npy', np.array(data))
np.save('../data/y_foreground_fold1_train.npy', np.array(label))

with open('evaluation_setup/fold2_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/foreground/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_foreground_fold2_train.npy', np.array(data))
np.save('../data/y_foreground_fold2_train.npy', np.array(label))

with open('evaluation_setup/fold3_train.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/foreground/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_foreground_fold3_train.npy', np.array(data))
np.save('../data/y_foreground_fold3_train.npy', np.array(label))


with open('evaluation_setup/fold4_train.txt') as f:
    lines = f.read().splitlines()


lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/foreground/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_foreground_fold4_train.npy', np.array(data))
np.save('../data/y_foreground_fold4_train.npy', np.array(label))

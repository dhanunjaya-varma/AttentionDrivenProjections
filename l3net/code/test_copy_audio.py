import shutil
import numpy as np


classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

with open('evaluation_setup/fold1_evaluate.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/audio/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_audio_fold1_test.npy', np.array(data))
np.save('../data/y_audio_fold1_test.npy', np.array(label))

with open('evaluation_setup/fold2_evaluate.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/audio/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_audio_fold2_test.npy', np.array(data))
np.save('../data/y_audio_fold2_test.npy', np.array(label))

with open('evaluation_setup/fold3_evaluate.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/audio/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_audio_fold3_test.npy', np.array(data))
np.save('../data/y_audio_fold3_test.npy', np.array(label))


with open('evaluation_setup/fold4_evaluate.txt') as f:
    lines = f.read().splitlines()


lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]
filename = [i.split('/', 1)[1] for i in src]

src_path = '../feat/audio/'

data = []
label = []
for i in range(len(filename)):
	f = filename[i].split('.')[0]
	src_pth = src_path+dest[i]+'/'+f+'.npy'
	data.append(np.mean(np.load(src_pth), axis=0))
	label.append(classes.index(dest[i]))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_audio_fold4_test.npy', np.array(data))
np.save('../data/y_audio_fold4_test.npy', np.array(label))

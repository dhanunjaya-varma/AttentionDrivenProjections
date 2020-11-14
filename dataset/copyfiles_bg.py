

with open('meta.txt') as f:
    lines = f.read().splitlines()

lines.sort()
src = [i.split('\t', 1)[0] for i in lines]
dest = [i.split('\t', 1)[1] for i in lines]
dest = [i.split('\t', 1)[0] for i in dest]

filename = [i.split('/', 1)[1] for i in src]

src_path = 'rpca_out_background/'
dest_path = 'background/'

import shutil
import os.path
for i in range(len(filename)):
    src_pth = src_path+filename[i]
    dst_pth = dest_path+dest[i]
    if os.path.isfile(src_pth):
        #print("file exists")
        shutil.copy(src_pth, dst_pth)
    else:
        print("file deosn't exists")
print(i)

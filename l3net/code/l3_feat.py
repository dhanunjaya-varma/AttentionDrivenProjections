#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:20:48 2019

@author: dhanunjaya
"""

import os
import numpy as np
import openl3
import soundfile as sf


classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']

input_path = '../../dataset/audio/'
output='../feat/audio/'


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        audio, sr = sf.read(filePath)
        #emb, ts = openl3.get_embedding(audio, sr)
        emb, ts = openl3.get_audio_embedding(audio, sr)
        outFileName = output + clas + '/' + file.split('.')[0]
        np.save(outFileName, emb)   


input_path = '../../dataset/background/'
output='../feat/background/'


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        audio, sr = sf.read(filePath)
        #emb, ts = openl3.get_embedding(audio, sr)
        emb, ts = openl3.get_audio_embedding(audio, sr)
        outFileName = output + clas + '/' + file.split('.')[0]
        np.save(outFileName, emb)

input_path = '../../dataset/foreground/'
output='../feat/foreground/'


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        audio, sr = sf.read(filePath)
        #emb, ts = openl3.get_embedding(audio, sr)
        emb, ts = openl3.get_audio_embedding(audio, sr)
        outFileName = output + clas + '/' + file.split('.')[0]
        np.save(outFileName, emb)

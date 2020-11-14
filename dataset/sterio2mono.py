#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:57:10 2019

@author: dhanunjaya
"""

import librosa
import os


files = os.listdir('stereo/')

for i in range(len(files)):
	filename = files[i];
	y, sr = librosa.load('stereo/'+filename, mono=False, sr=44100)
	y_mono = librosa.to_mono(y)
	librosa.output.write_wav('mono/'+filename, y_mono, sr)
print(i)

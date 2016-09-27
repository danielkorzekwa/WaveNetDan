'''
Created on Sep 27, 2016

@author: daniel
'''


import librosa
import numpy as np
import struct
from math import *
import pyglet
from time import sleep
import wave
import matplotlib.pyplot as plt

y = []
for i in range(0, 1000):
        value = 1*sin(i*(2*pi/3)) #+ random.randint(-4000, 4000)
        #packed_value = struct.pack('h', value)
        y.append(value)
    
    
librosa.output.write_wav('/home/daniel/daniel/sin_new.wav', np.array(y), 1000)

wave = librosa.load('/home/daniel/daniel/sin_new.wav',sr=1000)[0]
plt.plot(range(100),wave[0:100])
plt.show()

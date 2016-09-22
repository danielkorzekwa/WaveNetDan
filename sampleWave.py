'''
Created on Sep 21, 2016

@author: daniel
'''

from math import *
import struct

from WaveNetModel import WaveNetModel
import numpy as np


def sampleWave():
    '''Sample wave form.
    
    Returns np.array
    
    '''
    
    model = WaveNetModel()
    wave = []
    for i in range(0, 1000):
        sampleValue =  model.sample()
        print(sampleValue)
        wave.append(sampleValue)
        model.add(sampleValue)
    
    return np.array(wave)

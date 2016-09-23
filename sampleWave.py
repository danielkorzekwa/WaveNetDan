'''
Created on Sep 21, 2016

@author: daniel
'''

from WaveSampler import WaveSampler
import numpy as np
import time


def sampleWave(sampleNum):
    '''Sample wave form.
    
    Args:
        sampleNum (int): Number of samples to generate

    Returns:
        np.array: Generated waveform
        
    '''
    
    waveSampler = WaveSampler()
    wave = []
    for i in range(0, sampleNum):
        
        now = time.time()
        sampleValue =  waveSampler.sample()
        print(time.time()-now)
        
        print(sampleValue)
        wave.append(sampleValue)
    
    return np.array(wave)

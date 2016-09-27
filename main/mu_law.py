'''
Created on Sep 27, 2016

@author: daniel
'''

import numpy as np

def encode(wave, channels):
    '''
    
    Args
        wave(np.array[float]):
    
    Returns
        np.array[float]:
    '''
    mu = channels -1
    nom = np.log(1 + mu*np.abs(wave))
    denom = np.log(1+mu)
    s = np.sign(wave)*(nom/denom)
    return ((s + 1) / 2 * mu + 0.5).astype(np.int32)

def decode(encodedWave,channels):
    mu = channels - 1
    # Map values back to [-1, 1].
    casted = encodedWave.astype(np.float32)
    signal = 2 * (casted / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude  
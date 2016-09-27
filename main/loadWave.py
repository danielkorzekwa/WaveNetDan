'''
Created on Sep 26, 2016

@author: daniel
'''
import struct
import wave

import librosa

import numpy as np


def loadWave(waveFile,samplingRate):
    '''
    Loads waveform from file.
    
    Args
        waveFile(String): Waveform file name
        sampleRate(int):
    
    Returns
        np.array[int] Waveform amplitudes array
    '''
    s = librosa.load(waveFile,samplingRate)[0]
    return(s)
    
    
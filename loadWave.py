'''
Created on Sep 26, 2016

@author: daniel
'''
import wave
import struct

def loadWave(waveFile):
    '''
    Loads waveform from file.
    
    Args
        String: Waveform file name
    
    Returns
        np.array[int] Waveform amplitudes array
    '''
    s = wave.open(waveFile )

    y = []
    n=100
    for i in range(n):
        waveData = s.readframes(1)
        data = struct.unpack('<h',waveData)
        y.append(data[0])
    
    return y
    
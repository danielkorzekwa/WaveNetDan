'''
Created on Sep 21, 2016

@author: daniel
'''
import logging
import unittest

import librosa

from WaveGraph import WaveGraph
from loadWave import loadWave
import matplotlib.pyplot as plt
from sampleWave import sampleWave
import tensorflow as tf
from trainWave import trainWave


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

class test_sampleWave(unittest.TestCase):


    def test(self):
  
        waveform = loadWave('/home/daniel/daniel/o.wav', samplingRate=8000)        
        a = 0
        b = 500
               
       # notTrainedWave100 = sampleWave(waveform[0:500], 300)
       # f = plt.figure('not trained')
       # plt.plot(range(a, b), notTrainedWave100[a:b])
       # f.show()
       # input()
  
        logging.info('Training the model...') 
        filterOpParams = trainWave(waveform[0:1600], maxIterNum=200)
                
        trainedWave100 = sampleWave(waveform[0:250], 10,250,filterOpParams)
        plt.close()
        plt.plot(range(a, b), waveform[a:b],label='true')
        plt.plot(range(a, b), trainedWave100[a:b],label='predicted')
        plt.legend()
        plt.show()

        trainedWaveAll = sampleWave(waveform[0:4000], 10,4000,filterOpParams )
        librosa.output.write_wav('/home/daniel/daniel/test.wav', trainedWaveAll, 8000)

      

      
        
        input()

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

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
  
        waveform = loadWave('/home/daniel/daniel/sin_7000.wav', samplingRate=7000)        
        a = 450
        b = 750
        realPlot = plt.figure('real')
        plt.plot(range(a, b), waveform[a:b])
        realPlot.show()
          
        notTrainedWave100 = sampleWave(waveform[0:500], 300)
        f = plt.figure('not trained')
        plt.plot(range(a, b), notTrainedWave100[a:b])
        f.show()
  
        logging.info('Training the model...') 
        filterOpParams = trainWave(waveform[0:500], maxIterNum=2000)

        trainedWave100 = sampleWave(waveform[0:500], 300,filterOpParams)
        g = plt.figure('trained')
        plt.plot(range(a, b), trainedWave100[a:b])
        g.show()

        trainedWaveAll = sampleWave(waveform[0:3500], 3500,filterOpParams )
        librosa.output.write_wav('/home/daniel/daniel/test.wav', trainedWaveAll, 7000)

      

      
        
        input()

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

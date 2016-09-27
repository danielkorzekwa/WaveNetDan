'''
Created on Sep 21, 2016

@author: daniel
'''
import unittest

import librosa

from WaveGraph import WaveGraph
from loadWave import loadWave
from sampleWave import sampleWave
from trainWave import trainWave
import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.INFO)

class test_sampleWave(unittest.TestCase):


    def test(self):
  
        waveform = loadWave('/home/daniel/daniel/sin.wav', samplingRate=1000)        
        waveGraph = WaveGraph()

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        notTrainedWave = waveform #sampleWave(waveform[0:500], 500, waveGraph, sess)
  
        logging.info('Training the model...') 
        trainWave(waveform, waveGraph, sess)

        trainedWave = sampleWave(waveform[0:500], 500, waveGraph, sess)
        librosa.output.write_wav('/home/daniel/daniel/test.wav', trainedWave, 1000)

        a = 450
        b = 550
        realPlot = plt.figure('real')
        plt.plot(range(a, b), waveform[a:b])
        realPlot.show()

        f = plt.figure('not trained')
        plt.plot(range(a, b), notTrainedWave[a:b])
        f.show()
        g = plt.figure('trained')
        plt.plot(range(a, b), trainedWave[a:b])
        g.show()

        input()

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

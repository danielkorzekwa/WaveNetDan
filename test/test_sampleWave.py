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

class test_sampleWave(unittest.TestCase):


    def test(self):
       
        
        waveform = loadWave('/home/daniel/daniel/sin.wav')        
        
        waveGraph = WaveGraph()
        
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        
        trainWave(waveform,waveGraph,sess)

        wave = sampleWave(1000,waveGraph,sess)
        librosa.output.write_wav('/home/daniel/daniel/test.wav', wave, 1000)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
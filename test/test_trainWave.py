'''
Created on Sep 26, 2016

@author: daniel
'''
import unittest
from loadWave import loadWave
from trainWave import trainWave
import numpy as np

class test_trainWave(unittest.TestCase):


    def test(self):
        #waveform = loadWave('/home/daniel/daniel/sin.wav')
        x_data = np.arange(100).astype(np.float32)/100.0
        waveform = x_data * 0.1 + 0.3
        trainWave(waveform)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
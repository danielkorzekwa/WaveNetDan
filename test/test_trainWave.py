'''
Created on Sep 26, 2016

@author: daniel
'''
import unittest

from WaveGraph import WaveGraph
from loadWave import loadWave
import numpy as np
import tensorflow as tf
from trainWave import trainWave


class test_trainWave(unittest.TestCase):


    def test(self):
        waveform = loadWave('/home/daniel/daniel/sin_7000.wav',samplingRate=7000)        
                    
        trainWave(waveform,maxIterNum=50)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
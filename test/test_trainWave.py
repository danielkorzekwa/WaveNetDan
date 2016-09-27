'''
Created on Sep 26, 2016

@author: daniel
'''
import unittest
from loadWave import loadWave
from trainWave import trainWave
import numpy as np
from WaveGraph import WaveGraph

class test_trainWave(unittest.TestCase):


    def test(self):
        waveform = loadWave('/home/daniel/daniel/sin.wav')        
        
        waveGraph = WaveGraph()
        trainWave(waveform,waveGraph)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
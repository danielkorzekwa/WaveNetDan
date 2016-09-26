'''
Created on Sep 26, 2016

@author: daniel
'''
import unittest
from loadWave import loadWave
from trainWave import trainWave


class test_trainWave(unittest.TestCase):


    def test(self):
        waveform = loadWave('/home/daniel/daniel/sin.wav')
        trainWave(waveform)
    


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
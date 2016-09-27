'''
Created on Sep 26, 2016

@author: daniel
'''
import unittest
from loadWave import loadWave
import matplotlib.pyplot as plt

class test_loadWave(unittest.TestCase):


    def test(self):
        waveform = loadWave('/home/daniel/daniel/sin.wav')
        
        plt.plot(range(len(waveform)),waveform)
        plt.show()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
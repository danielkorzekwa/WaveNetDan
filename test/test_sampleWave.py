'''
Created on Sep 21, 2016

@author: daniel
'''
import unittest
from sampleWave import sampleWave
import librosa

def aaa(): return 4

class test_sampleWave(unittest.TestCase):


    def test(self):
        wave = sampleWave(500)
        librosa.output.write_wav('/home/daniel/daniel/test.wav', wave, 1000)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
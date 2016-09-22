'''
Created on Sep 21, 2016

@author: daniel
'''
from math import sin, pi
import tensorflow as tf
import numpy as np

class WaveNetModel:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.i = 0
        self.sess = tf.Session()
        self.normal = tf.truncated_normal([1], 0, 1000)
        
    def sample(self):
        '''Returns a single sample of a waveform
        '''
        # return int(20000*sin(self.i*(2*pi/3))) #+ random.randint(-4000, 4000)
        predictedOneHotProbs = np.ones(256) / 256
        predSample = np.random.choice(range(256), p=predictedOneHotProbs)
        return self.decode(predSample)
    
    def add(self, sample):
        '''Add sample to a model
        '''
        self.i += 1
    
    def decode(self, encodedSample):
        mu = 255
        y = encodedSample
        y = 2 * (y / mu) - 1
        x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
        return x
        

'''
Created on Sep 23, 2016

@author: daniel
'''

import tensorflow as tf
import numpy as np
import time

class WaveSampler:
    
    def __init__(self):
        '''
        Constructor
        '''
        self.sess = tf.Session()
        self.lastProbOp = self._createLastProbOp()
        
    def sample(self):
        '''Generate waveform amplitude for the next sample.
        
        @return float
        '''
        
        lastProb = self.sess.run(self.lastProbOp)
        predSample = np.random.choice(range(256), p=lastProb)
        return self._decode(predSample)
    
    def _createLastProbOp(self): 
        '''
        '''
        
        outputOp =  np.ones([10,256])
        
        outputProbsOp = tf.nn.softmax(outputOp)
        
        lastProbOp = tf.slice(outputProbsOp, 
                [tf.shape(outputProbsOp)[0]-1,0],[1,256])
        
        return tf.reshape(lastProbOp,[-1])

    def _decode(self,encodedSample):
        mu = 255
        y = encodedSample
        y = 2 * (y / mu) - 1
        x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
        return x      

        
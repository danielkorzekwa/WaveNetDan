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
      
        self.waveInput = tf.placeholder(tf.int32)
        self.lastProbOp = self._createLastProbOp()
        
    def sample(self,waveInput):
        '''Generate waveform amplitude for the next sample.
        
        Args
            waveInput(np.array(float))
        
        Returns
            float:
        '''
        if len(waveInput)== 0:
            return self._decode(np.random.randint(256))
        else:
            lastProb = self.sess.run(self.lastProbOp,feed_dict={self.waveInput:waveInput})
            print(lastProb)
            predSample = np.random.choice(range(256), p=lastProb)
            return self._decode(predSample)
    
    def _createLastProbOp(self): 
        '''
        '''

        waveInputOneHotOp = tf.one_hot(self.waveInput, 256)
        
        batchOneHotOp = tf.expand_dims(waveInputOneHotOp,0)
        
        filterOp = tf.truncated_normal([1,256,256],stddev=0.1)
        outputOp = tf.nn.conv1d(batchOneHotOp,filterOp,stride=1,padding='SAME')
         
        outputProbsOp = tf.nn.softmax(outputOp[0])
        
        lastProbOp = tf.slice(outputProbsOp, 
                [tf.shape(outputProbsOp)[0]-1,0],[1,256])
        
        return tf.reshape(lastProbOp,[-1])

    def _decode(self,encodedSample):
        mu = 255
        y = encodedSample
        y = 2 * (y / mu) - 1
        x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
        return x      

        
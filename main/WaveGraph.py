'''
Created on Sep 27, 2016

@author: daniel
'''

import tensorflow as tf


class WaveGraph:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
        self.waveInput = tf.placeholder(tf.int32)  # PLACEHOLDER [sample_num]
        
        self.waveInputOneHotOp = tf.one_hot(self.waveInput, 256)  # [sample_num,256]
    
        batchOneHotOp = tf.expand_dims(self.waveInputOneHotOp, 0)  # [1,sample_num,256]
        
        self.filterOp = tf.Variable(tf.truncated_normal([1, 256, 256], stddev=0.1))
        
        outputOp = tf.nn.conv1d(batchOneHotOp, self.filterOp, stride=1, padding='SAME')  # [1,sample_num,256]
         
        self.outputProbsOp = tf.nn.softmax(outputOp[0])  # [sample_num,256]
        
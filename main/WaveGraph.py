'''
Created on Sep 27, 2016

@author: daniel
'''

import tensorflow as tf


class WaveGraph:
    '''
    classdocs
    '''
    def __init__(self, filterOpParams=tf.truncated_normal([3, 256, 256], stddev=0.1)):
        '''
        
        Args
            filterOpParams(tensor_float32[1,256,256]):
        '''
        
        self.waveInput = tf.placeholder(tf.int32)  # [sample_num]
        
        self.waveInputOneHotOp = tf.one_hot(self.waveInput, 256)  # [sample_num,256]
    
        batchOneHotOp = tf.expand_dims(self.waveInputOneHotOp, 0)  # [1,sample_num,256]
        
        self.filterOp = tf.Variable(filterOpParams)
        
        outputOp1 = tf.nn.conv1d(batchOneHotOp, self.filterOp, stride=1, padding='SAME')  # [1,sample_num,256]
        outputOp2 = tf.nn.conv1d(outputOp1, self.filterOp, stride=1, padding='SAME')  # [1,sample_num,256]
        outputOp3 = tf.nn.conv1d(outputOp1, self.filterOp, stride=1, padding='SAME')  # [1,sample_num,256]
        self.outputProbsOp = tf.nn.softmax(outputOp3[0])  # [sample_num,256]
        

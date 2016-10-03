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
        
        self.filterParamsVar = tf.Variable(filterOpParams)
        
        outputOp1 = self._casualConv(batchOneHotOp,self.filterParamsVar)  # [1,sample_num,256]
        outputOp2 = self._casualConv(outputOp1,self.filterParamsVar)  # [1,sample_num,256]
        self.outputProbsOp = tf.nn.softmax(outputOp2[0])  # [sample_num,256]
     
    def _casualConv(self,waveOneHot,filterParamsVar):
        '''
        Args
            waveOneHot(tensor(1,sampleNum,256)):
            filterParamsVar(Variable):
            
        Returns tensor(1,sampleNum,256)
        '''
        
        
        padding = [[0, 0], [1, 0], [0, 0]]
        waveOneHotPad = tf.pad(waveOneHot,padding)
        
        convPad = tf.nn.conv1d(waveOneHotPad, filterParamsVar, stride=1, padding='SAME')  # [1,sample_num,256]
        casualConv =  tf.slice(convPad, [0, 0, 0], [-1, tf.shape(waveOneHot)[1], -1])
        
        return casualConv

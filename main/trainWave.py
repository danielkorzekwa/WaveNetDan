'''
Created on Sep 26, 2016

@author: daniel
'''
from WaveGraph import WaveGraph
import numpy as np
import tensorflow as tf


def trainWave(waveform,waveGraph,sess):
    ''' Trains wave model.
    
    Args:
        waveform(np.array(int)): Waveform array of amplitudes
        waveGraph(WaveGraph): Wave graph to train
    '''
        
    entropyOp = tf.reduce_mean(tf.reduce_sum(-waveGraph.waveInputOneHotOp * tf.log(waveGraph.outputProbsOp), reduction_indices=[1]))
        
    trainOp = tf.train.GradientDescentOptimizer(0.5).minimize(entropyOp)
        
    for i in range(2000):
        entropy =  sess.run([trainOp,entropyOp],feed_dict={waveGraph.waveInput:waveform})[1]
        print(entropy)
        #print(sess.run(waveTrainGraph.W),sess.run(waveTrainGraph.b))
      

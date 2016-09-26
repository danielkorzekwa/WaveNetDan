'''
Created on Sep 26, 2016

@author: daniel
'''
import tensorflow as tf
import numpy as np

def trainWave(waveform):
    ''' Trains wave model.
    
    Args:
        np.array(int) Waveform array of amplitudes
    '''
    
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    
    xData = np.arange(100).astype(np.float32)/100.0
    
    y = W * xData + b
    
    loss = tf.reduce_mean(tf.square(waveform.astype(np.float32)-y))
    
    train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i in range(200):
        sess.run(train)
        print(sess.run(loss))
        #print(sess.run(W),sess.run(b))
        
    raise Exception('Not implemented')     

'''
Created on Sep 21, 2016

@author: daniel
'''

import numpy as np
import tensorflow as tf

def sampleWave(sampleNum, waveGraph,sess):
    '''Sample wave form.
    
    Args:
        waveGraph (WaveGraph)
        sampleNum (int): Number of samples to generate

    Returns:
        np.array: Generated waveform
        
    '''
         
    lastProbOp = tf.slice(waveGraph.outputProbsOp, [tf.shape(waveGraph.outputProbsOp)[0] - 1, 0], [1, 256])
    lastProbReshapedOp = tf.reshape(lastProbOp,[-1])
    
    wave = []
    
    for i in range(0, 1000):
        
        if len(wave) == 0:
            wave.append(decode(np.random.randint(256)))
        else:
            lastProb = sess.run(lastProbReshapedOp, feed_dict={waveGraph.waveInput:wave})
            predSample = decode(np.random.choice(range(256), p=lastProb))
        
            print(predSample)
        
            wave.append(predSample)
    
    return np.array(wave)

def decode(encodedSample):
    mu = 255
    y = encodedSample
    y = 2 * (y / mu) - 1
    x = np.sign(y) * (1 / mu) * ((1 + mu) ** abs(y) - 1)
    return x   
